# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Any

import torch
import tyro
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING, DefaultDataCollator
from gr00t.utils.peft import get_lora_model


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    """Data configuration name from DATA_CONFIG_MAP, we assume all datasets have the same data config"""

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 8
    """Number of workers for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""

    # Training loop type
    use_trainer: bool = True
    """If False, run a custom manual training loop instead of HuggingFace Trainer."""


def _move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in a nested structure to the given device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        out = [_move_to_device(v, device) for v in batch]
        return type(batch)(out)
    return batch


def manual_train_loop(config: ArgsConfig, train_dataset: torch.utils.data.Dataset, model: torch.nn.Module) -> None:
    """Lightweight single-GPU training loop that bypasses HuggingFace Trainer."""
    assert (
        config.num_gpus == 1
    ), "manual_train_loop currently supports only single-GPU training. Use num_gpus=1."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device is required for GR00T fine-tuning.")

    model.to(device)
    model.train()

    # DataLoader closely mirrors the Trainer settings
    # NOTE: For diagnosis, we disable shuffle so that sampling pattern is deterministic
    # and heavy data-loading paths (e.g., new trajectories) are easier to reason about.
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        collate_fn=DefaultDataCollator(),
        pin_memory=True,
        persistent_workers=config.dataloader_num_workers > 0,
        drop_last=True,
    )

    # Optimizer with decoupled weight decay (same structure as DualBrainTrainer)
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(0.95, 0.999),
        eps=1e-8,
    )

    total_steps = config.max_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    os.makedirs(config.output_dir, exist_ok=True)
    writer: SummaryWriter | None = None
    if config.report_to == "tensorboard":
        writer = SummaryWriter(log_dir=config.output_dir)

    global_step = 0
    epoch = 0

    print("\n" + "=" * 70)
    print("🚀 STARTING MANUAL GR00T TRAINING LOOP (no HF Trainer)")
    print("=" * 70)
    print(
        f"batch_size={config.batch_size}, "
        f"num_workers={config.dataloader_num_workers}, "
        f"max_steps={config.max_steps}, "
        f"save_steps={config.save_steps}"
    )
    print("=" * 70 + "\n")

    # Use an explicit iterator so we can time data loading separately
    dataloader_iter = iter(dataloader)

    while global_step < total_steps:
        epoch += 1
        while global_step < total_steps:
            step_wall_start = time.time()
            # ---- Data loading ----
            t0 = time.time()
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # New epoch
                epoch += 1
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            t_data = time.time() - t0

            # ---- CPU -> GPU transfer ----
            t0 = time.time()
            batch = _move_to_device(batch, device)
            torch.cuda.synchronize()
            t_transfer = time.time() - t0

            # ---- Forward ----
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(batch)
                loss = outputs["loss"]
            torch.cuda.synchronize()
            t_forward = time.time() - t0

            # ---- Backward ----
            t0 = time.time()
            loss.backward()
            torch.cuda.synchronize()
            t_backward = time.time() - t0

            # ---- Optimizer + Scheduler ----
            t0 = time.time()
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            t_optim = time.time() - t0

            global_step += 1
            step_wall = time.time() - step_wall_start

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/step_time_s", step_wall, global_step)
                writer.add_scalar("train/time_data_s", t_data, global_step)
                writer.add_scalar("train/time_transfer_s", t_transfer, global_step)
                writer.add_scalar("train/time_forward_s", t_forward, global_step)
                writer.add_scalar("train/time_backward_s", t_backward, global_step)
                writer.add_scalar("train/time_optim_s", t_optim, global_step)

            if global_step == 1 or global_step <= 10 or global_step % 10 == 0:
                print(
                    f"[MANUAL] step={global_step} "
                    f"loss={loss.item():.4f} "
                    f"total={step_wall:.3f}s "
                    f"data={t_data:.3f}s "
                    f"transfer={t_transfer:.3f}s "
                    f"forward={t_forward:.3f}s "
                    f"backward={t_backward:.3f}s "
                    f"optim={t_optim:.3f}s",
                    flush=True,
                )

            if config.save_steps > 0 and global_step % config.save_steps == 0:
                ckpt_dir = os.path.join(config.output_dir, f"checkpoint-step-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                # Save only the tuned GR00T weights (same as Trainer.save_model)
                model.save_pretrained(ckpt_dir)
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    os.path.join(ckpt_dir, "optimizer_sched.pt"),
                )
                print(f"[MANUAL] Saved checkpoint to {ckpt_dir}", flush=True)

            if global_step >= total_steps:
                break

    if writer is not None:
        writer.close()

    print("\n" + "=" * 70)
    print(f"✅ MANUAL TRAINING COMPLETE. total_steps={global_step}")
    print("=" * 70 + "\n")


#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader: we will use either single dataset or mixture dataset
    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            ## We use the same transforms, modality configs, and embodiment tag for all datasets here,
            ## in reality, you can use dataset from different modalities and embodiment tags
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # we will use equal weights for all datasets
                for dataset in single_datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets, with {config.dataset_path} ")

    # ------------ step 2: load model ------------
    # First, get the data config to determine action horizon
    data_action_horizon = len(data_config_cls.action_indices)

    # Load model
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
    )

    # Update action_horizon to match data config
    # Need to recreate action head with correct config since it was initialized with old config
    if data_action_horizon != model.action_head.config.action_horizon:
        print(
            f"Recreating action head with action_horizon {data_action_horizon} (was {model.action_head.config.action_horizon})"
        )

        # Update the action head config
        new_action_head_config = model.action_head.config
        new_action_head_config.action_horizon = data_action_horizon

        # Import the FlowmatchingActionHead class
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        # Create new action head with updated config
        new_action_head = FlowmatchingActionHead(new_action_head_config)

        # Copy the weights from the old action head to the new one
        new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

        # Replace the action head
        model.action_head = new_action_head

        # Update model config AND the action_head_cfg dictionary that gets saved
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon

        # Set trainable parameters for the new action head
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector, tune_diffusion_model=config.tune_diffusion_model
        )

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # 2.1 modify training args
    if config.use_trainer:
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
            dataloader_pin_memory=True,  # 빠른 CPU→GPU 전송
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

        # 2.2 run experiment (HF Trainer path)
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    experiment.train()
    else:
        # Pure PyTorch loop without HuggingFace Trainer
        manual_train_loop(config, train_dataset, model)


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
            ]

            # Convert config to command line arguments
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    cmd.append(f"--{key.replace('_', '-')}")

                    # if the value is a list (e.g. dataset_path), we need to add each element in the list
                    if isinstance(value, list):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
