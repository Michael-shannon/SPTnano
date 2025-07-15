"""Test feature calculation module."""

import numpy as np
import pandas as pd
import pytest

from SPTnano.features import ParticleMetrics


@pytest.fixture()
def sample_trajectory_data():
    """Create sample trajectory data for testing."""
    # Create a simple circular trajectory
    n_frames = 100
    t = np.linspace(0, 2 * np.pi, n_frames)
    radius = 1.0

    data = {
        "unique_id": ["track_001"] * n_frames,
        "frame": range(n_frames),
        "x": radius * np.cos(t),
        "y": radius * np.sin(t),
        "time_s": np.arange(n_frames) * 0.01,
        "condition": ["test_condition"] * n_frames,
        "filename": ["test_file.tiff"] * n_frames,
    }

    return pd.DataFrame(data)


@pytest.fixture()
def straight_trajectory_data():
    """Create a simple straight trajectory for testing."""
    n_frames = 50

    data = {
        "unique_id": ["track_002"] * n_frames,
        "frame": range(n_frames),
        "x": np.linspace(0, 5, n_frames),  # Moving in straight line
        "y": np.zeros(n_frames),
        "time_s": np.arange(n_frames) * 0.01,
        "condition": ["test_condition"] * n_frames,
        "filename": ["test_file.tiff"] * n_frames,
    }

    return pd.DataFrame(data)


class TestParticleMetrics:
    """Test ParticleMetrics class functionality."""

    def test_initialization(self, sample_trajectory_data):
        """Test ParticleMetrics initialization."""
        metrics = ParticleMetrics(sample_trajectory_data, time_between_frames=0.01)

        assert metrics.df is not None
        assert len(metrics.df) == len(sample_trajectory_data)
        assert metrics.time_between_frames == 0.01
        assert hasattr(metrics, "tolerance")

    def test_distance_calculation(self, straight_trajectory_data):
        """Test distance calculation between consecutive frames."""
        metrics = ParticleMetrics(straight_trajectory_data, time_between_frames=0.01)
        metrics.calculate_distances()

        # For straight line movement, distances should be roughly constant
        distances = metrics.df["distance_um"].dropna()
        assert len(distances) > 0
        assert all(distances >= 0)  # Distances should be non-negative

    def test_speed_calculation(self, sample_trajectory_data):
        """Test speed calculation."""
        metrics = ParticleMetrics(sample_trajectory_data, time_between_frames=0.01)
        metrics.calculate_distances()
        metrics.calculate_speeds()

        speeds = metrics.df["speed_um_s"].dropna()
        assert len(speeds) > 0
        assert all(speeds >= 0)  # Speeds should be non-negative

    def test_direction_calculation(self, sample_trajectory_data):
        """Test direction calculation."""
        metrics = ParticleMetrics(sample_trajectory_data, time_between_frames=0.01)
        metrics.calculate_directions()

        directions = metrics.df["direction_rad"].dropna()
        assert len(directions) > 0
        # Directions should be in range [-π, π]
        assert all(directions >= -np.pi)
        assert all(directions <= np.pi)

    def test_acceleration_calculation(self, sample_trajectory_data):
        """Test acceleration calculation."""
        metrics = ParticleMetrics(sample_trajectory_data, time_between_frames=0.01)
        metrics.calculate_distances()
        metrics.calculate_speeds()
        metrics.calculate_accelerations()

        accelerations = metrics.df["acceleration_um_s2"].dropna()
        assert len(accelerations) > 0
        # Should have finite values
        assert all(np.isfinite(accelerations))

    def test_feature_calculation_pipeline(self, sample_trajectory_data):
        """Test full feature calculation pipeline."""
        metrics = ParticleMetrics(sample_trajectory_data, time_between_frames=0.01)

        # Run basic feature calculation (without time-windowed and time-averaged)
        result_df = metrics.calculate_all_features(
            calculate_time_windowed=False, calculate_time_averaged=False
        )

        assert result_df is not None
        assert len(result_df) == len(sample_trajectory_data)

        # Check that key features were calculated
        expected_columns = [
            "distance_um",
            "speed_um_s",
            "direction_rad",
            "acceleration_um_s2",
            "jerk_um_s3",
        ]

        for col in expected_columns:
            assert col in result_df.columns

    def test_msd_calculation_for_track(self, sample_trajectory_data):
        """Test MSD calculation for a single track."""
        metrics = ParticleMetrics(sample_trajectory_data, time_between_frames=0.01)

        # Add required coordinate columns
        metrics.df["x_um"] = metrics.df["x"]
        metrics.df["y_um"] = metrics.df["y"]

        # Test MSD calculation
        avg_msd, D, alpha, motion_class = metrics.calculate_msd_for_track(
            metrics.df, allow_partial_window=True, min_window_size=10
        )

        assert avg_msd > 0
        assert D > 0  # Diffusion coefficient should be positive
        assert alpha > 0  # Anomalous exponent should be positive
        assert motion_class in ["normal", "subdiffusive", "superdiffusive"]

    def test_extract_condition_info(self, sample_trajectory_data):
        """Test extraction of condition information."""
        metrics = ParticleMetrics(sample_trajectory_data, time_between_frames=0.01)

        # Test location extraction
        location = metrics.extract_location("Condition_freehalo_cort")
        assert location == "cort"

        # Test molecule extraction
        molecule = metrics.extract_molecule("Condition_freehalo_cort")
        assert molecule == "freehalo"

    def test_tolerance_calculation(self, sample_trajectory_data):
        """Test tolerance calculation method."""
        metrics = ParticleMetrics(sample_trajectory_data, time_between_frames=0.01)
        tolerance = metrics.calculate_tolerance()

        assert isinstance(tolerance, (int, float))
        assert tolerance > 0
