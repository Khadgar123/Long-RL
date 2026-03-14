#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen3-4B  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.ulysses_size=2 \
    data.val_batch_size=32 \
    data.rollout_batch_size=32 \
    worker.actor.global_batch_size=32 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_4b_math_grpo \
    trainer.logger="console,tensorboard"  # 添加这行
