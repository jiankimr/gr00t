# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Validation module for GR00T training
# This module can be safely removed without affecting the core training code.
# To disable validation: simply don't import this module or set --enable_validation=False

"""
Validation Module for GR00T Training

This module provides validation functionality that can be easily enabled/disabled.
It does NOT affect training performance when disabled.

Usage:
    from gr00t.experiment.validation import (
        create_train_val_split,
        ValidationCallback,
        add_validation_args
    )

To disable: Simply don't use these functions or set enable_validation=False
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


@dataclass
class ValidationConfig:
    """Configuration for validation."""
    enable_validation: bool = False
    val_split_ratio: float = 0.05  # 5% for validation (e.g., 50 episodes -> 2-3 for val)
    eval_steps: int = 100  # Evaluate every N steps
    eval_on_start: bool = False  # Whether to evaluate before training
    val_seed: int = 42  # Seed for reproducible validation split


def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """
    Split dataset into train and validation sets.
    
    This uses a deterministic split based on seed, so the same split
    is always produced for the same dataset and seed.
    
    Args:
        dataset: The full dataset to split
        val_ratio: Ratio of data to use for validation (default: 0.1 = 10%)
        seed: Random seed for reproducible split
        
    Returns:
        Tuple of (train_subset, val_subset)
    """
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    # Create deterministic indices
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Sort indices to maintain some ordering (optional, helps with caching)
    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    print(f"[Validation] Dataset split: {train_size} train, {val_size} val ({val_ratio*100:.1f}%)")
    
    return train_subset, val_subset


class ValidationCallback(TrainerCallback):
    """
    Callback for logging validation metrics.
    
    This callback is optional and doesn't affect training when not used.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.val_history = []
        
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ):
        """Called after evaluation."""
        # Log validation metrics
        val_metrics = {
            "step": state.global_step,
            "epoch": state.epoch,
        }
        
        # Extract eval metrics
        for key, value in metrics.items():
            if key.startswith("eval_"):
                val_metrics[key] = value
                
        self.val_history.append(val_metrics)
        
        # Save validation history
        history_path = self.output_dir / "validation_history.json"
        with open(history_path, "w") as f:
            json.dump(self.val_history, f, indent=2)
            
        # Print summary
        if "eval_loss" in metrics:
            print(f"\n[Validation] Step {state.global_step}: val_loss = {metrics['eval_loss']:.4f}")
            
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of training."""
        if self.val_history:
            print(f"\n[Validation] Training complete. {len(self.val_history)} evaluations performed.")
            print(f"[Validation] History saved to: {self.output_dir / 'validation_history.json'}")


def setup_validation_args(
    training_args: TrainingArguments,
    val_config: ValidationConfig,
) -> TrainingArguments:
    """
    Modify TrainingArguments to enable validation.
    
    This only modifies eval-related settings and doesn't affect
    training hyperparameters like learning rate, batch size, etc.
    
    Args:
        training_args: Original TrainingArguments
        val_config: Validation configuration
        
    Returns:
        Modified TrainingArguments with validation enabled
    """
    if not val_config.enable_validation:
        return training_args
    
    # Enable evaluation
    training_args.do_eval = True
    # Use eval_strategy (newer) and evaluation_strategy (older) for compatibility
    training_args.eval_strategy = "steps"
    training_args.evaluation_strategy = "steps"  # For older versions
    training_args.eval_steps = val_config.eval_steps
    training_args.eval_on_start = val_config.eval_on_start
    
    # Use same batch size for eval (can be adjusted if memory is an issue)
    training_args.per_device_eval_batch_size = training_args.per_device_train_batch_size
    
    print(f"[Validation] Enabled: eval every {val_config.eval_steps} steps")
    
    return training_args


def add_validation_to_trainer(trainer, val_dataset: Dataset, output_dir: str):
    """
    Add validation dataset and callback to an existing trainer.
    
    Args:
        trainer: The HuggingFace Trainer instance
        val_dataset: Validation dataset
        output_dir: Directory to save validation logs
    """
    trainer.eval_dataset = val_dataset
    trainer.add_callback(ValidationCallback(output_dir))
    print(f"[Validation] Added validation dataset with {len(val_dataset)} samples")


# Utility function to check if validation is properly set up
def validate_setup(trainer, val_config: ValidationConfig) -> bool:
    """Check if validation is properly configured."""
    if not val_config.enable_validation:
        return True
        
    if trainer.eval_dataset is None:
        print("[Validation] Warning: Validation enabled but no eval_dataset set!")
        return False
        
    if not trainer.args.do_eval:
        print("[Validation] Warning: Validation enabled but do_eval is False!")
        return False
        
    return True

