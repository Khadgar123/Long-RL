# Adopted from https://github.com/hiyouga/EasyR1. Below is the original copyright:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
基于 Ray 单控制器的 FSDP PPO 训练器。
该训练器支持使用 huggingface 模型进行模型无关的初始化。

核心架构:
- Ray 单控制器: 驱动进程通过 RPC 调用 Worker Group 的计算函数
- Worker Group: 分布在多个 GPU 上的 Worker 进程组
- 角色: Actor(训练), Rollout(推理), Critic(价值函数), RefPolicy(参考策略)
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .core_algos import AdvantageEstimator, FixedKLController, KLController, compute_kl, get_kl_controller
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics


class Role(IntEnum):
    """
    定义分布式训练中的 Worker 角色。
    可以在单 GPU 上共存多个角色以节省显存（colocated）。

    角色说明:
    - Actor: 策略模型，负责训练更新
    - Rollout: 生成模型，负责采样 responses
    - ActorRollout: 合并的 Actor + Rollout
    - Critic: 价值函数，用于 GAE  advantage 估计
    - RefPolicy: 参考策略，用于计算 KL 散度
    - RewardModel: 奖励模型
    - ActorRolloutRef: 合并的 Actor + Rollout + Ref（典型配置）
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    资源池管理器，定义分布式训练的资源分配规格。
    每个资源池包含多个 GPU 节点，每个节点有若干 GPU。

    示例:
        resource_pool_spec = {
            "pool1": [8],     # 1个节点，8个GPU
            "pool2": [4, 4],  # 2个节点，每个4个GPU
        }
    """

    resource_pool_spec: dict[str, list[int]]  # 资源池规格: {pool_name: [每节点GPU数]}
    mapping: dict[Role, str]                   # 角色到资源池的映射
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """
    应用 KL 惩罚，将 KL 散度作为负奖励加入最终奖励。

    作用:
    - 防止策略更新过快，保持与参考策略的相似性
    - KL 散度越大，惩罚越大

    公式:
        token_level_rewards = token_level_scores - kl_coef * KL(old || ref)
    """
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # 计算参考策略和当前策略之间的 KL 散度
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # 只计算 response 部分的 KL

    # 奖励 = 原始分数 - KL系数 * KL散度
    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    # 计算平均 KL（用于监控）
    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # 动态调整 KL 系数（自适应控制）
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """
    计算 Advantage（优势）和 Return（回报）。

    Advantage 估计算法:
    - GAE: Generalized Advantage Estimation，需要 Critic 预测的 value
    - GRPO: Group Relative Policy Optimization，同一 prompt 生成的多个 response 之间比较
    - REINFORCE++: 增强版 REINFORCE，使用 gamma 折扣作为 baseline
    - RLOO: REINFORCE with Leave-One-Out Baseline，使用其他样本的均值作为 baseline
    - REMAX: 使用 max(response rewards) 作为 baseline
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        # GAE: 需要 Critic 预测的价值函数
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # GRPO: 按组归一化，同 prompt 的 response 为一组
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        # REINFORCE++: 使用 gamma 折扣作为 baseline
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        # REMAX: 使用 batch 中最大 reward 作为 baseline
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        # RLOO: Leave-One-Out，使用其他样本均值作为 baseline
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data

def compute_advantage_diffusion(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    token_level_rewards = data.batch["token_level_rewards"] # (batch_size, 1)
    if adv_estimator == AdvantageEstimator.GRPO:
        batch_size = token_level_rewards.shape[0]
        response_mask = torch.ones((batch_size, 1), dtype=torch.bool)
        index = torch.zeros((batch_size), dtype=torch.long)
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    基于 Ray 的 PPO 训练器。

    特点:
    - 单控制器架构: 驱动进程在 CPU 上，负责调度
    - Worker Group: 多个 GPU 进程执行实际计算
    - 混合引擎: Actor + Rollout + Ref 合并部署在同 GPU（hybrid_engine=True）
    - 支持多种 RL 算法: PPO, GRPO, DAPO, REINFORCE++, RLOO, REMAX
    - 支持多模态: 图像、视频、音频理解

    典型训练流程:
    1. 生成 (Rollout): 使用当前策略生成 responses
    2. 奖励 (Reward): 使用奖励函数计算每个 token 的奖励
    3. 价值 (Value): 使用 Critic 预测价值（仅 GAE）
    4. Advantage: 计算 advantage 和 return
    5. 更新 Critic: 训练价值函数（仅 GAE）
    6. 更新 Actor: 使用 PPO/GRPO 等算法更新策略
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        self.diffusion = config.trainer.diffusion

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        step_file = os.path.join(self.config.trainer.load_checkpoint_path, "checkpoint_tracker.json")
        if os.path.exists(step_file):
            load_step = json.load(open(step_file))["last_global_step"]
            self.config.trainer.load_checkpoint_path = os.path.join(self.config.trainer.load_checkpoint_path, "global_step_%d"%load_step)

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        """
        验证流程。

        步骤:
        1. 初始化空的奖励列表
        2. 准备 rollout engine（vLLM）
        3. 遍历验证集，生成 responses
        4. 使用奖励函数计算奖励
        5. 记录验证样本到日志（用于分析）
        6. 释放 rollout engine
        7. 返回验证指标
        """
        reward_tensor_lst = []
        # 用于收集验证样本（在日志中展示）
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        print("Start validation...")

        if not self.diffusion:
            self.actor_rollout_ref_wg.prepare_rollout_engine()
        print("***len(val_dataloader)***", len(self.val_dataloader))
        for idx, batch_dict in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(batch_dict)

            if self.diffusion:
                # For diffusion models, pop embedding-related keys
                test_gen_batch = test_batch.pop(
                    batch_keys=["prompt_embeds", "pooled_prompt_embeds", "negative_prompt_embeds",
                                "negative_pooled_prompt_embeds"]
                )
                repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
                test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
                repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
                test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
                test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
                test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels

            if not self.diffusion:
                test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
                test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)
            else:
                # For diffusion, directly generate without padding/unpadding
                test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)

            # repeat to align with repeated responses in rollout
            if not self.diffusion:
                test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            if not self.diffusion:
                input_ids = test_batch.batch["prompts"]
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                output_ids = test_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            else:
                # For diffusion, handle differently based on what's available
                # Assume prompt_text and generated images are available in the batch
                if "text" in test_batch.non_tensor_batch:
                    input_texts = test_batch.non_tensor_batch["text"].tolist()
                else:
                    # Fallback to placeholder if no text prompts available
                    input_texts = [f"Diffusion prompt {i}" for i in range(len(test_batch))]

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_scores.extend(scores)
            if not self.diffusion:
                sample_outputs.extend(output_texts)
                sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        if not self.diffusion:
            self.actor_rollout_ref_wg.release_rollout_engine()

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics}

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """
        序列长度负载均衡。

        问题: 不同样本的序列长度差异很大，如果直接按顺序分配给各 GPU，
              会导致部分 GPU 很忙，部分 GPU 很闲。

        解决方案:
        - 计算每个样本的总 token 数
        - 按 token 数排序，使用贪心算法分配给各 GPU
        - 使每个 GPU 处理的总 token 数尽可能均衡

        注意: 这会打乱样本顺序，需要注意 GRPO/RLOO 等按组计算的算法。
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        # 计算每个样本的总 token 数
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_ref_wg.world_size
        # 均衡分区，每个 GPU 分配到的样本总 token 数接近
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # 按索引重新排序，数据会自动被 dispatch 到各 GPU
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        # 记录负载均衡统计信息
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: Dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {"min_pixels": self.config.data.min_pixels, "max_pixels": self.config.data.max_pixels}
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            if self.diffusion:
                gen_batch = new_batch.pop(batch_keys=["prompt_embeds", "pooled_prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"])
            else:
                # pop those keys for generation
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "ground_truth"],
                    meta_info_keys=["min_pixels", "max_pixels"],
                )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            # 为每个 prompt 生成 n 个 responses（n = rollout.n）
            # repeat n 次后，每个原始 prompt 会有 n 个对应的 response
            if not self.diffusion:
                new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # 在线过滤：根据奖励过滤低质量样本
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                # 按 uid（原始 prompt）分组，计算每组平均分
                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                # 只保留平均分在 [filter_low, filter_high] 范围内的样本
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) > 0:
                    new_batch = new_batch[kept_sample_idxs]

            # 拼接多个 mini-batch（可能需要多次采样才能达到目标 batch size）
            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise ValueError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        PPO 训练循环。

        驱动进程只需要通过 RPC 调用 Worker Group 的计算函数来构建 PPO 数据流。
        轻量级的 advantage 计算在驱动进程上完成。

        完整训练流程:
        ┌─────────────────────────────────────────────────────────┐
        │  1. _make_batch_data()     - 采样数据，生成 responses   │
        │          ↓                                            │
        │  2. _balance_batch()       - 序列长度负载均衡           │
        │          ↓                                            │
        │  3. compute_reward()       - 计算奖励                  │
        │          ↓                                            │
        │  4. compute_log_probs()     - 计算旧策略的 log_prob     │
        │          ↓                                            │
        │  5. compute_ref_log_probs()- 计算参考策略的 log_prob   │
        │          ↓                                            │
        │  6. compute_values()       - 计算价值（仅 GAE）        │
        │          ↓                                            │
        │  7. compute_advantage()    - 计算 advantage 和 return  │
        │          ↓                                            │
        │  8. update_critic()        - 更新 Critic（仅 GAE）      │
        │          ↓                                            │
        │  9. update_actor()         - 更新 Actor                │
        │          ↓                                            │
        │ 10. _validate()            - 验证（可选）              │
        │          ↓                                            │
        │ 11. _save_checkpoint()     - 保存 checkpoint           │
        └─────────────────────────────────────────────────────────┘
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # ===== 1. 生成 (Rollout): 使用当前策略生成 responses =====
                with timer("gen", timing_raw):
                    if not self.diffusion:
                        # 准备 vLLM 推理引擎
                        self.actor_rollout_ref_wg.prepare_rollout_engine()
                        batch = self._make_batch_data(metrics=metrics)
                        # 释放 vLLM 引擎，释放显存
                        self.actor_rollout_ref_wg.release_rollout_engine()
                    else:
                        batch = self._make_batch_data(metrics=metrics)

                # ===== 2. 序列长度负载均衡 =====
                # NOTE: 这会打乱样本顺序，需要注意 GRPO/RLOO 等按组计算的算法
                if not self.diffusion:
                    self._balance_batch(batch, metrics=metrics)
                    # 记录全局有效 token 数
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # ===== 3. 计算奖励 =====
                if not self.diffusion:
                    if "token_level_scores" not in batch.batch:
                        with timer("reward", timing_raw):
                            # 异步计算奖励
                            reward_ref = self.reward_fn.compute_reward.remote(batch)
                else:
                    reward_ref = self.reward_fn.compute_reward.remote(batch)

                # ===== 4. 重新计算旧策略的 log_prob =====
                if not self.diffusion:
                    with timer("old", timing_raw):
                        # 计算生成 responses 时的 log probability（用于 PPO 裁剪）
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                # ===== 5. 计算参考策略的 log_prob（用于 KL 惩罚）=====
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # ===== 6. 计算价值（仅 GAE 算法需要）=====
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                # ===== 7. 计算 Advantage 和 Return =====
                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # 异步获取奖励
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # 应用 KL 惩罚（如果启用）
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # 计算 advantage，在驱动进程上执行（轻量级）
                    compute_advantage_func = compute_advantage_diffusion if self.diffusion else compute_advantage
                    batch = compute_advantage_func(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                # ===== 8. 更新 Critic（仅 GAE 算法）=====
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # ===== 9. 更新 Actor（策略模型）=====
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # ===== 10. 验证 =====
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                # ===== 11. 保存 Checkpoint =====
                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, diffusion=self.diffusion))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw, diffusion=self.diffusion))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus, diffusion=self.diffusion))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
