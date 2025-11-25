"""
Noise transform for action data.
Based on LIBERO_GROOT_TRAINING_GUIDE.md Section 5.3 / Appendix C
"""

import numpy as np
from gr00t.data.transform.base import ModalityTransform


class ActionNoiseTransform(ModalityTransform):
    """
    Adds time-series noise to specific action dimensions.
    
    This transform is used for the "Noisy Version" training to study
    sabotage/stealth degradation scenarios where task success is maintained
    while secondary metrics (jerk, energy consumption, etc.) are degraded.
    
    Args:
        apply_to (list[str]): List of action keys to apply noise to
        target_dim (int): Dimension index to apply noise to (default: 0)
        pattern (list[float]): Noise pattern to repeat (default: [1.0, 1.0, -1.0, -1.0])
        amplitude (float): Noise amplitude multiplier (default: 0.25)
        clip_range (tuple[float, float]): Range to clip values to (default: (-1.0, 1.0))
    
    Example:
        For action.eef_pos_delta with target_dim=0, pattern=[1.0, 1.0, -1.0, -1.0],
        and amplitude=0.3:
        - Generates: [0.3, 0.3, -0.3, -0.3, 0.3, 0.3, -0.3, -0.3, ...]
        - Adds this to action_data[:, 0] (x-axis movement)
        - Creates oscillating pattern in x-axis movement
    """

    target_dim: int = 0
    pattern: list[float] = [1.0, 1.0, -1.0, -1.0]
    amplitude: float = 0.25
    clip_range: tuple[float, float] = (-1.0, 1.0)

    def apply(self, data: dict, metadata: dict = None) -> dict:
        """Apply noise transform to action data."""
        for key in self.apply_to:
            if key in data:
                action_data = data[key]  # Shape: [horizon, action_dim]
                
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
                    self.clip_range[1]
                )
                
                data[key] = action_data
                
        return data
    
    def __call__(self, data: dict, metadata: dict = None) -> dict:
        """Backward compatibility wrapper."""
        return self.apply(data, metadata)


__all__ = ["ActionNoiseTransform"]

