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

from typing import Any

import numpy as np
import torch
from pydantic import Field

from gr00t.data.transform.base import ModalityTransform


class ActionNoiseTransform(ModalityTransform):
    """
    Adds time-series noise to specific action dimensions.
    """

    target_dim: int = Field(..., description="The dimension index to add noise to.")
    pattern: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, -1.0, -1.0],
        description="The noise pattern to repeat (e.g., [1.0, 1.0, -1.0, -1.0] for ++-- pattern).",
    )
    amplitude: float = Field(default=0.25, description="The amplitude of the noise to add.")
    clip_range: tuple[float, float] = Field(
        default=(-1.0, 1.0), description="The range to clip the noisy values to."
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply noise pattern to action data."""
        if not self.training:
            # Don't add noise in eval mode
            return data

        for key in self.apply_to:
            if key in data:
                action_data = data[key]  # Shape: [horizon, action_dim]

                # Convert to numpy if it's a torch tensor
                is_torch = isinstance(action_data, torch.Tensor)
                if is_torch:
                    action_data = action_data.cpu().numpy()

                # Generate noise pattern for the entire horizon
                horizon_length = action_data.shape[0]
                pattern_length = len(self.pattern)

                # Repeat pattern to cover the horizon
                num_repeats = (horizon_length + pattern_length - 1) // pattern_length
                extended_pattern = (self.pattern * num_repeats)[:horizon_length]

                # Create noise array
                noise = np.array(extended_pattern, dtype=action_data.dtype) * self.amplitude

                # Add noise to target dimension
                action_data[:, self.target_dim] += noise

                # Clip to specified range
                action_data[:, self.target_dim] = np.clip(
                    action_data[:, self.target_dim],
                    self.clip_range[0],
                    self.clip_range[1],
                )

                # Convert back to torch if it was originally a torch tensor
                if is_torch:
                    action_data = torch.from_numpy(action_data).to(
                        data[key].device if hasattr(data[key], "device") else "cpu"
                    )

                data[key] = action_data

        return data


class StochasticActionNoiseTransform(ModalityTransform):
    """
    Adds time-series noise to specific action dimensions with a given probability.
    This allows training on a mix of clean and noisy data.
    """

    target_dim: int = Field(..., description="The dimension index to add noise to.")
    pattern: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, -1.0, -1.0],
        description="The noise pattern to repeat (e.g., [1.0, 1.0, -1.0, -1.0] for ++-- pattern).",
    )
    amplitude: float = Field(default=0.25, description="The amplitude of the noise to add.")
    clip_range: tuple[float, float] = Field(
        default=(-1.0, 1.0), description="The range to clip the noisy values to."
    )
    noise_probability: float = Field(
        default=0.5, description="Probability of applying noise to each sample."
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply noise pattern to action data with given probability."""
        if not self.training:
            # Don't add noise in eval mode
            return data

        # Randomly decide whether to add noise to this sample
        if np.random.random() > self.noise_probability:
            return data

        for key in self.apply_to:
            if key in data:
                action_data = data[key]  # Shape: [horizon, action_dim]

                # Convert to numpy if it's a torch tensor
                is_torch = isinstance(action_data, torch.Tensor)
                if is_torch:
                    action_data = action_data.cpu().numpy()

                # Generate noise pattern for the entire horizon
                horizon_length = action_data.shape[0]
                pattern_length = len(self.pattern)

                # Repeat pattern to cover the horizon
                num_repeats = (horizon_length + pattern_length - 1) // pattern_length
                extended_pattern = (self.pattern * num_repeats)[:horizon_length]

                # Create noise array
                noise = np.array(extended_pattern, dtype=action_data.dtype) * self.amplitude

                # Add noise to target dimension
                action_data[:, self.target_dim] += noise

                # Clip to specified range
                action_data[:, self.target_dim] = np.clip(
                    action_data[:, self.target_dim],
                    self.clip_range[0],
                    self.clip_range[1],
                )

                # Convert back to torch if it was originally a torch tensor
                if is_torch:
                    action_data = torch.from_numpy(action_data).to(
                        data[key].device if hasattr(data[key], "device") else "cpu"
                    )

                data[key] = action_data

        return data
