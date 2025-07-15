"""
Advanced augmentation strategies for single-molecule trajectory data
"""

import random

import numpy as np
import torch


class TrajectoryAugmentations:
    """
    Collection of biologically-motivated augmentations for trajectory data
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        time_warp_sigma: float = 0.2,
        magnitude_warp_sigma: float = 0.2,
        rotation_angle_range: float = 0.1,
        scaling_range: tuple[float, float] = (0.9, 1.1),
        temporal_mask_ratio: float = 0.1,
    ):
        """
        Initialize augmentation parameters

        Parameters
        ----------
        noise_std : float
            Standard deviation for Gaussian noise
        time_warp_sigma : float
            Sigma for time warping (temporal distortion)
        magnitude_warp_sigma : float
            Sigma for magnitude warping (spatial distortion)
        rotation_angle_range : float
            Range for random rotation (radians)
        scaling_range : tuple
            Min/max scaling factors
        temporal_mask_ratio : float
            Fraction of time points to mask

        """
        self.noise_std = noise_std
        self.time_warp_sigma = time_warp_sigma
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.rotation_angle_range = rotation_angle_range
        self.scaling_range = scaling_range
        self.temporal_mask_ratio = temporal_mask_ratio

    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to trajectory
        Simulates measurement noise in microscopy
        """
        noise = torch.randn_like(x, device=x.device) * self.noise_std
        return x + noise

    def time_warping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping to trajectory
        Simulates variable frame rates or temporal irregularities
        """
        B, T, D = x.shape

        # Generate smooth time warping curve
        warp_steps = torch.cumsum(
            torch.exp(torch.randn(T, device=x.device) * self.time_warp_sigma), dim=0
        )
        warp_steps = warp_steps / warp_steps[-1] * (T - 1)

        # Interpolate trajectory at warped time points
        warped_x = torch.zeros_like(x)
        for b in range(B):
            for d in range(D):
                warped_x[b, :, d] = torch.nn.functional.interpolate(
                    x[b, :, d].unsqueeze(0).unsqueeze(0),
                    size=T,
                    mode="linear",
                    align_corners=True,
                ).squeeze()

        return warped_x

    def magnitude_warping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply magnitude warping to trajectory
        Simulates variable diffusion coefficients or local environment changes
        """
        B, T, D = x.shape

        # Generate smooth magnitude scaling curve
        magnitude_curve = torch.exp(
            torch.cumsum(
                torch.randn(T, device=x.device) * self.magnitude_warp_sigma, dim=0
            )
        )
        magnitude_curve = magnitude_curve / magnitude_curve.mean()

        # Apply magnitude warping
        warped_x = x.clone()
        for d in range(D):
            warped_x[:, :, d] *= magnitude_curve.unsqueeze(0)

        return warped_x

    def rotation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random rotation to trajectory
        Simulates different orientations of the same motion pattern
        Note: Only applies to spatial dimensions (dx, dy)
        """
        if x.shape[-1] < 2:
            return x  # Need at least 2D for rotation

        B, T, D = x.shape
        angle = (torch.rand(B, device=x.device) - 0.5) * 2 * self.rotation_angle_range

        # Create rotation matrices
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        rotated_x = x.clone()

        # Apply rotation to dx, dy (first two dimensions)
        for b in range(B):
            dx = x[b, :, 0]
            dy = x[b, :, 1]
            rotated_x[b, :, 0] = dx * cos_a[b] - dy * sin_a[b]
            rotated_x[b, :, 1] = dx * sin_a[b] + dy * cos_a[b]

        return rotated_x

    def scaling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random scaling to trajectory
        Simulates different diffusion rates or measurement scales
        """
        B = x.shape[0]
        scale_factor = (
            torch.rand(B, device=x.device)
            * (self.scaling_range[1] - self.scaling_range[0])
            + self.scaling_range[0]
        )

        scaled_x = x.clone()
        for b in range(B):
            scaled_x[b] *= scale_factor[b]

        return scaled_x

    def temporal_masking(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly mask temporal segments
        Simulates missing frames or detection failures
        """
        B, T, D = x.shape
        masked_x = x.clone()

        for b in range(B):
            n_mask = int(T * self.temporal_mask_ratio)
            if n_mask > 0:
                mask_start = torch.randint(
                    0, T - n_mask + 1, (1,), device=x.device
                ).item()
                # Replace with interpolated values instead of zeros
                if mask_start > 0 and mask_start + n_mask < T:
                    for d in range(D):
                        start_val = masked_x[b, mask_start - 1, d]
                        end_val = masked_x[b, mask_start + n_mask, d]
                        interp_vals = torch.linspace(
                            start_val, end_val, n_mask + 2, device=x.device
                        )[1:-1]
                        masked_x[b, mask_start : mask_start + n_mask, d] = interp_vals

        return masked_x

    def velocity_perturbation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add correlated noise to velocity components
        Simulates systematic measurement errors or drift
        """
        B, T, D = x.shape

        # Generate correlated noise for velocity components
        if D >= 2:  # dx, dy available
            # Create correlated noise
            correlation = 0.3  # Moderate correlation between dx and dy noise
            noise_dx = torch.randn(B, T, device=x.device) * self.noise_std
            noise_dy = (
                correlation * noise_dx
                + np.sqrt(1 - correlation**2) * torch.randn(B, T, device=x.device)
            ) * self.noise_std

            perturbed_x = x.clone()
            perturbed_x[:, :, 0] += noise_dx  # dx
            perturbed_x[:, :, 1] += noise_dy  # dy

            return perturbed_x

        return self.gaussian_noise(x)

    def direction_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add small perturbations to direction component
        Simulates angular measurement uncertainty
        """
        if x.shape[-1] < 3:
            return x  # No direction component

        jittered_x = x.clone()
        direction_noise = (
            torch.randn_like(x[:, :, 2], device=x.device) * 0.05
        )  # Small angular noise
        jittered_x[:, :, 2] += direction_noise

        # Keep direction in valid range [-π, π]
        jittered_x[:, :, 2] = (
            torch.remainder(jittered_x[:, :, 2] + np.pi, 2 * np.pi) - np.pi
        )

        return jittered_x

    def apply_augmentation(
        self, x: torch.Tensor, augmentation_type: str = "random"
    ) -> torch.Tensor:
        """
        Apply specified augmentation or random selection

        Parameters
        ----------
        x : torch.Tensor
            Input trajectory [B, T, D]
        augmentation_type : str
            Type of augmentation to apply
            Options: 'noise', 'time_warp', 'magnitude_warp', 'rotation',
                    'scaling', 'temporal_mask', 'velocity_pert', 'direction_jitter', 'random'

        Returns
        -------
        torch.Tensor
            Augmented trajectory

        """
        if augmentation_type == "random":
            # Randomly select augmentation
            augmentations = [
                "noise",
                "time_warp",
                "magnitude_warp",
                "rotation",
                "scaling",
                "temporal_mask",
                "velocity_pert",
                "direction_jitter",
            ]
            augmentation_type = random.choice(augmentations)

        if augmentation_type == "noise":
            return self.gaussian_noise(x)
        elif augmentation_type == "time_warp":
            return self.time_warping(x)
        elif augmentation_type == "magnitude_warp":
            return self.magnitude_warping(x)
        elif augmentation_type == "rotation":
            return self.rotation(x)
        elif augmentation_type == "scaling":
            return self.scaling(x)
        elif augmentation_type == "temporal_mask":
            return self.temporal_masking(x)
        elif augmentation_type == "velocity_pert":
            return self.velocity_perturbation(x)
        elif augmentation_type == "direction_jitter":
            return self.direction_jitter(x)
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")

    def compose_augmentations(
        self,
        x: torch.Tensor,
        augmentations: list[str],
        probabilities: list[float] | None = None,
    ) -> torch.Tensor:
        """
        Apply multiple augmentations with specified probabilities

        Parameters
        ----------
        x : torch.Tensor
            Input trajectory
        augmentations : List[str]
            List of augmentation types to apply
        probabilities : List[float], optional
            Probability of applying each augmentation (default: 0.5 for all)

        Returns
        -------
        torch.Tensor
            Augmented trajectory

        """
        if probabilities is None:
            probabilities = [0.5] * len(augmentations)

        augmented_x = x.clone()

        for aug_type, prob in zip(augmentations, probabilities, strict=False):
            if torch.rand(1, device=x.device).item() < prob:
                augmented_x = self.apply_augmentation(augmented_x, aug_type)

        return augmented_x


# Predefined augmentation strategies
class AugmentationStrategies:
    """
    Predefined augmentation strategies for different use cases
    """

    @staticmethod
    def basic_noise(noise_std: float = 0.01):
        """Basic Gaussian noise (current implementation)"""
        aug = TrajectoryAugmentations(noise_std=noise_std)
        return lambda x: aug.gaussian_noise(x)

    @staticmethod
    def measurement_noise():
        """Simulates realistic measurement noise"""
        aug = TrajectoryAugmentations(noise_std=0.015)
        return lambda x: aug.compose_augmentations(
            x, ["noise", "velocity_pert", "direction_jitter"], [0.8, 0.6, 0.4]
        )

    @staticmethod
    def temporal_variations():
        """Simulates temporal irregularities"""
        aug = TrajectoryAugmentations(time_warp_sigma=0.1, temporal_mask_ratio=0.05)
        return lambda x: aug.compose_augmentations(
            x, ["time_warp", "temporal_mask"], [0.5, 0.3]
        )

    @staticmethod
    def spatial_variations():
        """Simulates spatial transformations"""
        aug = TrajectoryAugmentations(
            rotation_angle_range=0.2, scaling_range=(0.85, 1.15)
        )
        return lambda x: aug.compose_augmentations(
            x, ["rotation", "scaling", "magnitude_warp"], [0.6, 0.4, 0.3]
        )

    @staticmethod
    def comprehensive():
        """Comprehensive augmentation strategy"""
        aug = TrajectoryAugmentations(
            noise_std=0.012,
            time_warp_sigma=0.15,
            magnitude_warp_sigma=0.1,
            rotation_angle_range=0.15,
            scaling_range=(0.9, 1.1),
            temporal_mask_ratio=0.08,
        )
        return lambda x: aug.compose_augmentations(
            x, ["noise", "rotation", "scaling", "velocity_pert"], [0.7, 0.4, 0.3, 0.5]
        )

    @staticmethod
    def conservative():
        """Conservative augmentation for sensitive data"""
        aug = TrajectoryAugmentations(
            noise_std=0.008, rotation_angle_range=0.05, scaling_range=(0.95, 1.05)
        )
        return lambda x: aug.compose_augmentations(
            x, ["noise", "direction_jitter"], [0.6, 0.3]
        )


def get_augmentation_function(strategy: str = "basic"):
    """
    Get augmentation function by strategy name

    Parameters
    ----------
    strategy : str
        Augmentation strategy name
        Options: 'basic', 'measurement_noise', 'temporal_variations',
                'spatial_variations', 'comprehensive', 'conservative'

    Returns
    -------
    callable
        Augmentation function

    """
    strategies = {
        "basic": AugmentationStrategies.basic_noise,
        "measurement_noise": AugmentationStrategies.measurement_noise,
        "temporal_variations": AugmentationStrategies.temporal_variations,
        "spatial_variations": AugmentationStrategies.spatial_variations,
        "comprehensive": AugmentationStrategies.comprehensive,
        "conservative": AugmentationStrategies.conservative,
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}"
        )

    return strategies[strategy]()
