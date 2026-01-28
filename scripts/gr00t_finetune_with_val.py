#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GR00T Fine-tuning with Validation Support
#
# This script extends gr00t_finetune.py with validation loss tracking.
# It can be easily reverted by simply using the original gr00t_finetune.py
#
# Usage:
#   python gr00t_finetune_with_val.py --enable_validation --val_split 0.1 --eval_steps 100 ...
#
# To disable validation (same behavior as original):
#   python gr00t_finetune_with_val.py ...  (--enable_validation defaults to False)

"""
GR00T Fine-tuning with Validation Support

This script adds validation loss tracking without affecting training performance.
When validation is disabled, behavior is identical to the original gr00t_finetune.py.

Key features:
- Validation split is deterministic (same seed = same split)
- Validation does NOT affect training (no gradient updates during eval)
- Easy to enable/disable via command line flag
- Validation history saved to JSON for analysis
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.experiment.validation import (
    ValidationCallback,
    ValidationConfig,
    create_train_val_split,
    setup_validation_args,
)
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning with validation support."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: str = "fourier_gr1_arms_only"
    """Data configuration to use for training."""

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
    """Whether to use the full model for LORA."""

    dataloader_num_workers: int = 12
    """Number of workers for data loading per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training."""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    """Video backend to use for training."""

    balance_dataset_weights: bool = True
    """Balance dataset weights in mixture."""

    balance_trajectory_weights: bool = True
    """Balance trajectory weights within dataset."""

    # ============ VALIDATION PARAMETERS ============
    # These parameters control validation behavior.
    # When enable_validation=False, training is identical to original.
    
    enable_validation: bool = False
    """Enable validation loss tracking. Default: False (same as original)."""

    val_split: float = 0.05
    """Fraction of data to use for validation (default: 0.05 = 5%, ~2-3 episodes for 50 ep dataset)."""

    eval_steps: int = 100
    """Evaluate every N training steps."""

    eval_on_start: bool = False
    """Run evaluation before training starts."""

    val_seed: int = 42
    """Random seed for validation split (deterministic)."""


#####################################################################################
# Helper functions (copied from original gr00t_finetune.py)
#####################################################################################


def _copy_partial_action_expert_weights(old_dict, new_dict, old_dim, new_dim):
    """Copy weights with partial dimension matching for action_dim changes."""
    total_params = copied_params = random_params = 0

    for key, old_tensor in old_dict.items():
        if key not in new_dict:
            continue

        new_tensor = new_dict[key]
        total_params += new_tensor.numel()

        if old_tensor.shape == new_tensor.shape:
            new_tensor.copy_(old_tensor)
            copied_params += new_tensor.numel()
        elif "action_encoder" in key and "W1.weight" in key:
            new_tensor[:, :old_dim] = old_tensor
            copied_params += old_tensor.numel()
            random_params += new_tensor.numel() - old_tensor.numel()
        elif "action_decoder" in key and ("weight" in key or "bias" in key):
            if old_tensor.dim() == 1:
                new_tensor[:old_dim] = old_tensor
            elif old_tensor.dim() == 2:
                new_tensor[:, :old_dim] = old_tensor
            elif old_tensor.dim() == 3:
                new_tensor[:, :, :old_dim] = old_tensor
            copied_params += old_tensor.numel()
            random_params += new_tensor.numel() - old_tensor.numel()
        else:
            random_params += new_tensor.numel()

    assert total_params == copied_params + random_params, "Parameter count mismatch"
    random_percentage = (random_params / total_params) * 100 if total_params > 0 else 0
    print(
        f"Weight copy stats: {copied_params:,} copied, {random_params:,} random ({random_percentage:.1f}% randomly initialized)"
    )
    print(f"Action dimensions {old_dim+1}-{new_dim} will be learned from scratch")
    return new_dict


#####################################################################################
# Custom Trainer with Validation Support
#####################################################################################


class DualBrainTrainerWithEval:
    """
    Custom trainer that properly handles GR00T model evaluation.
    GR00T model has a different forward signature than standard HuggingFace models.
    """
    
    @staticmethod
    def create(base_trainer_class):
        """Create a custom trainer class that supports GR00T evaluation."""
        from typing import Optional, Union, Tuple, Dict, List, Any
        
        class CustomTrainer(base_trainer_class):
            def prediction_step(
                self,
                model,
                inputs,
                prediction_loss_only: bool,
                ignore_keys: Optional[List[str]] = None,
            ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
                """
                Custom prediction step for GR00T models.
                GR00T uses model(inputs) instead of model(**inputs).
                """
                model.eval()
                
                with torch.no_grad():
                    # GR00T expects inputs as a dict, not unpacked kwargs
                    outputs = model(inputs)
                    loss = outputs.get("loss", None)
                
                # Return (loss, logits, labels) format expected by Trainer
                # For GR00T, we only return loss as we don't have standard logits/labels
                return (loss, None, None)
        
        return CustomTrainer


#####################################################################################
# Extended TrainRunner with Validation Support
#####################################################################################


class TrainRunnerWithValidation(TrainRunner):
    """Extended TrainRunner that supports validation with Subset datasets."""
    
    def create_trainer_with_eval(
        self,
        model,
        training_args,
        train_dataset,
        eval_dataset,
        data_collator,
        compute_dtype,
    ):
        """Create trainer with optional eval_dataset support."""
        from gr00t.experiment.trainer import DualBrainTrainer
        from gr00t.utils.experiment import CheckpointFormatCallback
        
        # Use custom trainer that properly handles GR00T evaluation
        if eval_dataset is not None:
            CustomTrainer = DualBrainTrainerWithEval.create(DualBrainTrainer)
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_dtype=compute_dtype,
            )
        else:
            # Use original trainer if no eval
            trainer = DualBrainTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_dtype=compute_dtype,
            )
        
        # Add checkpoint format callback
        run_name = training_args.run_name
        ckpt_format_callback = CheckpointFormatCallback(
            run_name=run_name, exp_cfg_dir=self.exp_cfg_dir
        )
        trainer.add_callback(ckpt_format_callback)
        
        # Log dataloader information
        train_dl_len = len(trainer.get_train_dataloader())
        print(
            f"train dataloader length: {train_dl_len}\n"
            f"train dataset length: {len(trainer.train_dataset)}\n"
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer
    
    def __init__(
        self,
        model,
        training_args,
        train_dataset,
        eval_dataset=None,
        resume_from_checkpoint=False,
        val_config=None,
        original_dataset=None,  # Original dataset for metadata
    ):
        from torch.utils.data import Subset
        
        # Store for later use
        self._eval_dataset = eval_dataset
        self._val_config = val_config
        self._original_dataset = original_dataset
        
        # If train_dataset is a Subset, we need to handle metadata specially
        # Extract original dataset for metadata
        if isinstance(train_dataset, Subset) and original_dataset is None:
            original_dataset = train_dataset.dataset
            self._original_dataset = original_dataset
        
        # Override the metadata writing logic by temporarily replacing the dataset
        # We'll write metadata from the original dataset, not the Subset
        self.training_args = training_args
        self.output_dir = Path(training_args.output_dir)
        self.exp_cfg_dir = self.output_dir / "experiment_cfg"
        self.exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset
        
        # Set up run name
        training_args.run_name = (
            training_args.output_dir.split("/")[-1]
            if training_args.run_name is None
            else training_args.run_name
        )
        print(f"Run name: {training_args.run_name}")
        
        from gr00t.model.transforms import DefaultDataCollator
        data_collator = DefaultDataCollator()
        
        from transformers import set_seed
        compute_dtype = torch.float16 if training_args.bf16 else torch.float32
        set_seed(training_args.seed)
        
        # Create trainer with eval_dataset if validation is enabled
        trainer = self.create_trainer_with_eval(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if (val_config and val_config.enable_validation) else None,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )
        self.trainer = trainer
        
        # Write metadata using the original dataset (not Subset)
        self.rank = int(os.environ.get("RANK", 0))
        if self.rank == 0:
            metadata_json = {}
            if os.path.exists(self.exp_cfg_dir / "metadata.json"):
                with open(self.exp_cfg_dir / "metadata.json", "r") as f:
                    metadata_json = json.load(f)
            
            # Use original dataset for metadata
            metadata_source = original_dataset if original_dataset else train_dataset
            
            if isinstance(metadata_source, LeRobotSingleDataset):
                metadata_json.update(
                    {metadata_source.tag: metadata_source.metadata.model_dump(mode="json")}
                )
            elif isinstance(metadata_source, LeRobotMixtureDataset):
                metadata_json.update(
                    {
                        tag: metadata.model_dump(mode="json")
                        for tag, metadata in metadata_source.merged_metadata.items()
                    }
                )
            elif isinstance(metadata_source, Subset):
                # Handle nested Subset
                inner_dataset = metadata_source.dataset
                if isinstance(inner_dataset, LeRobotSingleDataset):
                    metadata_json.update(
                        {inner_dataset.tag: inner_dataset.metadata.model_dump(mode="json")}
                    )
            # Skip metadata writing if we can't determine the type
            
            with open(self.exp_cfg_dir / "metadata.json", "w") as f:
                json.dump(metadata_json, f, indent=4)
        
        # Set up reporting
        report_to = training_args.report_to
        if report_to == "wandb":
            if "WANDB_PROJECT" not in os.environ:
                os.environ["WANDB_PROJECT"] = "gr00t-training"
            if "WANDB_RUN_ID" not in os.environ:
                runtime_id = os.environ.get("RUNTIME_ID", None)
                if runtime_id:
                    os.environ["WANDB_RUN_ID"] = runtime_id
            os.environ["WANDB_DIR"] = training_args.output_dir
            
            wandb_config_file = self.output_dir / "wandb_config.json"
            with open(wandb_config_file, "w") as f:
                json.dump(
                    {
                        "project": os.environ.get("WANDB_PROJECT", ""),
                        "run_id": os.environ.get("WANDB_RUN_ID", ""),
                    },
                    f,
                )
            training_args.report_to = ["wandb"]
        elif report_to == "azure_ml":
            print("azure_ml logging is enabled.")
        else:
            tensorboard_dir = Path(training_args.output_dir) / "runs"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
            training_args.report_to = ["tensorboard"]
        
        # Add validation callback (eval_dataset already passed to trainer)
        if eval_dataset is not None and val_config and val_config.enable_validation:
            self.trainer.add_callback(ValidationCallback(training_args.output_dir))
            print(f"[Validation] Added eval dataset with {len(eval_dataset)} samples")


#####################################################################################
# Main training function with validation
#####################################################################################


def main(config: ArgsConfig):
    """Main training function with validation support."""
    
    # Create validation config
    val_config = ValidationConfig(
        enable_validation=config.enable_validation,
        val_split_ratio=config.val_split,
        eval_steps=config.eval_steps,
        eval_on_start=config.eval_on_start,
        val_seed=config.val_seed,
    )
    
    if val_config.enable_validation:
        print("\n" + "=" * 50)
        print("🔍 VALIDATION ENABLED")
        print("=" * 50)
        print(f"  Val split ratio: {val_config.val_split_ratio * 100:.1f}%")
        print(f"  Eval every: {val_config.eval_steps} steps")
        print(f"  Val seed: {val_config.val_seed}")
        print("=" * 50 + "\n")
    
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # Load full dataset first
    if len(config.dataset_path) == 1:
        full_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
            single_datasets.append(dataset)

        full_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)
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
        print(f"Loaded {len(single_datasets)} datasets")

    # Split dataset if validation is enabled
    if val_config.enable_validation:
        train_dataset, eval_dataset = create_train_val_split(
            full_dataset,
            val_ratio=val_config.val_split_ratio,
            seed=val_config.val_seed,
        )
    else:
        train_dataset = full_dataset
        eval_dataset = None

    # ------------ step 2: load model ------------
    data_action_horizon = len(data_config_cls.action_indices)

    assert (
        hasattr(transforms, "transforms") and len(transforms.transforms) > 0
    ), "No transforms found"
    last_transform = transforms.transforms[-1]
    from gr00t.model.transforms import GR00TTransform

    assert isinstance(last_transform, GR00TTransform), "Last transform must be GR00TTransform"
    assert hasattr(last_transform, "max_action_dim"), "GR00TTransform must have max_action_dim"
    data_max_action_dim = last_transform.max_action_dim

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )

    # Handle action dimension mismatch
    action_horizon_mismatch = data_action_horizon != model.action_head.config.action_horizon
    action_dim_mismatch = data_max_action_dim != model.action_head.config.action_dim

    if action_horizon_mismatch or action_dim_mismatch:
        old_action_horizon = model.action_head.config.action_horizon
        old_action_dim = model.action_head.config.action_dim
        print(
            f"Recreating action head with action_horizon {data_action_horizon} (was {old_action_horizon})"
        )
        if action_dim_mismatch:
            print(f"Updating max_action_dim {data_max_action_dim} (was {old_action_dim})")

        import copy

        new_action_head_config = copy.deepcopy(model.action_head.config)
        new_action_head_config.action_horizon = data_action_horizon
        new_action_head_config.action_dim = data_max_action_dim

        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        new_action_head = FlowmatchingActionHead(new_action_head_config)

        if not action_dim_mismatch:
            print("Copying weights from old action head (compatible dimensions)")
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
        else:
            print(
                f"Partial weight copy: copying first {old_action_dim} dimensions, initializing last {data_max_action_dim - old_action_dim} dimensions randomly"
            )
            new_action_head.state_dict().update(
                _copy_partial_action_expert_weights(
                    model.action_head.state_dict(),
                    new_action_head.state_dict(),
                    old_action_dim,
                    data_max_action_dim,
                )
            )

        model.action_head = new_action_head

        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon
        model.config.action_head_cfg["action_dim"] = data_max_action_dim

        model.config.action_dim = data_max_action_dim
        model.action_dim = data_max_action_dim

        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector, tune_diffusion_model=config.tune_diffusion_model
        )

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

    # ------------ step 3: setup training arguments ------------
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
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
        save_total_limit=5,
        report_to=config.report_to,
        seed=42,
        do_eval=False,  # Will be modified if validation enabled
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # Apply validation settings if enabled
    training_args = setup_validation_args(training_args, val_config)

    # ------------ step 4: run experiment ------------
    experiment = TrainRunnerWithValidation(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
        val_config=val_config,
        original_dataset=full_dataset,  # Pass original for metadata
    )

    experiment.train()


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)

    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING WITH VALIDATION SUPPORT")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) > available ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            script_path = Path(__file__).absolute()
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
                *raw_args_list,
            ]

            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)

