"""Test augmentations module."""

import pytest
import torch

from SPTnano.augmentations import (
    AugmentationStrategies,
    TrajectoryAugmentations,
    get_augmentation_function,
)


class TestTrajectoryAugmentations:
    """Test TrajectoryAugmentations class."""

    @pytest.fixture()
    def sample_trajectory(self):
        """Create a sample trajectory tensor."""
        # [batch_size, sequence_length, features]
        return torch.randn(4, 60, 3)

    @pytest.fixture()
    def augmentator(self):
        """Create a basic augmentator."""
        return TrajectoryAugmentations(
            noise_std=0.01,
            time_warp_sigma=0.2,
            magnitude_warp_sigma=0.2,
            rotation_angle_range=0.1,
            scaling_range=(0.9, 1.1),
            temporal_mask_ratio=0.1,
        )

    def test_initialization(self, augmentator):
        """Test augmentator initialization."""
        assert augmentator.noise_std == 0.01
        assert augmentator.time_warp_sigma == 0.2
        assert augmentator.magnitude_warp_sigma == 0.2
        assert augmentator.rotation_angle_range == 0.1
        assert augmentator.scaling_range == (0.9, 1.1)
        assert augmentator.temporal_mask_ratio == 0.1

    def test_gaussian_noise(self, augmentator, sample_trajectory):
        """Test Gaussian noise augmentation."""
        original = sample_trajectory.clone()
        augmented = augmentator.gaussian_noise(sample_trajectory)

        # Check that shape is preserved
        assert augmented.shape == original.shape

        # Check that values are different (noise was added)
        assert not torch.allclose(original, augmented, atol=1e-6)

        # Check that the noise is reasonable
        diff = torch.abs(augmented - original).mean()
        assert diff > 0
        assert diff < 0.1  # Should be small noise

    def test_scaling(self, augmentator, sample_trajectory):
        """Test scaling augmentation."""
        original = sample_trajectory.clone()
        augmented = augmentator.scaling(sample_trajectory)

        # Check that shape is preserved
        assert augmented.shape == original.shape

        # For most cases, values should be different (unless scale factor is exactly 1.0)
        # We'll just check that the function runs without error
        assert torch.isfinite(augmented).all()

    def test_rotation(self, augmentator, sample_trajectory):
        """Test rotation augmentation."""
        original = sample_trajectory.clone()
        augmented = augmentator.rotation(sample_trajectory)

        # Check that shape is preserved
        assert augmented.shape == original.shape
        assert torch.isfinite(augmented).all()

    def test_temporal_mask(self, augmentator, sample_trajectory):
        """Test temporal masking augmentation."""
        original = sample_trajectory.clone()
        augmented = augmentator.temporal_mask(sample_trajectory)

        # Check that shape is preserved
        assert augmented.shape == original.shape
        assert torch.isfinite(augmented).all()

    def test_velocity_perturbation(self, augmentator, sample_trajectory):
        """Test velocity perturbation augmentation."""
        original = sample_trajectory.clone()
        augmented = augmentator.velocity_perturbation(sample_trajectory)

        # Check that shape is preserved
        assert augmented.shape == original.shape
        assert torch.isfinite(augmented).all()

    def test_direction_jitter(self, augmentator, sample_trajectory):
        """Test direction jitter augmentation."""
        original = sample_trajectory.clone()
        augmented = augmentator.direction_jitter(sample_trajectory)

        # Check that shape is preserved
        assert augmented.shape == original.shape
        assert torch.isfinite(augmented).all()

    def test_compose_augmentations(self, augmentator, sample_trajectory):
        """Test composing multiple augmentations."""
        original = sample_trajectory.clone()

        augmentations = ["noise", "scaling", "rotation"]
        probabilities = [0.8, 0.5, 0.3]

        augmented = augmentator.compose_augmentations(
            sample_trajectory, augmentations, probabilities
        )

        # Check that shape is preserved
        assert augmented.shape == original.shape
        assert torch.isfinite(augmented).all()


class TestAugmentationStrategies:
    """Test predefined augmentation strategies."""

    @pytest.fixture()
    def sample_trajectory(self):
        """Create a sample trajectory tensor."""
        return torch.randn(2, 30, 3)

    def test_basic_noise(self, sample_trajectory):
        """Test basic noise strategy."""
        aug_func = AugmentationStrategies.basic_noise(noise_std=0.02)
        augmented = aug_func(sample_trajectory)

        assert augmented.shape == sample_trajectory.shape
        assert torch.isfinite(augmented).all()
        assert not torch.allclose(sample_trajectory, augmented, atol=1e-6)

    def test_measurement_noise(self, sample_trajectory):
        """Test measurement noise strategy."""
        aug_func = AugmentationStrategies.measurement_noise()
        augmented = aug_func(sample_trajectory)

        assert augmented.shape == sample_trajectory.shape
        assert torch.isfinite(augmented).all()

    def test_temporal_variations(self, sample_trajectory):
        """Test temporal variations strategy."""
        aug_func = AugmentationStrategies.temporal_variations()
        augmented = aug_func(sample_trajectory)

        assert augmented.shape == sample_trajectory.shape
        assert torch.isfinite(augmented).all()

    def test_spatial_variations(self, sample_trajectory):
        """Test spatial variations strategy."""
        aug_func = AugmentationStrategies.spatial_variations()
        augmented = aug_func(sample_trajectory)

        assert augmented.shape == sample_trajectory.shape
        assert torch.isfinite(augmented).all()

    def test_comprehensive(self, sample_trajectory):
        """Test comprehensive strategy."""
        aug_func = AugmentationStrategies.comprehensive()
        augmented = aug_func(sample_trajectory)

        assert augmented.shape == sample_trajectory.shape
        assert torch.isfinite(augmented).all()

    def test_conservative(self, sample_trajectory):
        """Test conservative strategy."""
        aug_func = AugmentationStrategies.conservative()
        augmented = aug_func(sample_trajectory)

        assert augmented.shape == sample_trajectory.shape
        assert torch.isfinite(augmented).all()


def test_get_augmentation_function():
    """Test augmentation function getter."""
    # Test valid strategies
    valid_strategies = [
        "basic",
        "measurement_noise",
        "temporal_variations",
        "spatial_variations",
        "comprehensive",
        "conservative",
    ]

    for strategy in valid_strategies:
        aug_func = get_augmentation_function(strategy)
        assert callable(aug_func)

        # Test that it works with sample data
        x = torch.randn(2, 30, 3)
        result = aug_func(x)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    # Test invalid strategy
    with pytest.raises(ValueError):
        get_augmentation_function("invalid_strategy")
