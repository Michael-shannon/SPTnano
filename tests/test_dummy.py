"""Test package import and basic functionality."""

import SPTnano


def test_package_import() -> None:
    """Test that the package can be imported successfully."""
    assert SPTnano is not None
    assert hasattr(SPTnano, "ParticleMetrics")
    assert hasattr(SPTnano, "config")


def test_main_classes_importable() -> None:
    """Test that main classes can be imported."""
    from SPTnano import ParticleMetrics
    from SPTnano.augmentations import TrajectoryAugmentations
    from SPTnano.transformer import TransformerMotionEncoder

    # Just test that they're classes
    assert callable(ParticleMetrics)
    assert callable(TransformerMotionEncoder)
    assert callable(TrajectoryAugmentations)
