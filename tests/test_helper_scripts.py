"""Test helper scripts module."""

import pytest
import pandas as pd
import numpy as np
from SPTnano.helper_scripts import filter_stubs, pathfixer


@pytest.fixture
def sample_tracking_data():
    """Create sample tracking data for testing."""
    data = []
    
    # Create tracks of different lengths
    track_lengths = [5, 10, 20, 50, 100]  # frames
    time_between_frames = 0.1  # seconds
    
    for i, length in enumerate(track_lengths):
        track_data = {
            'unique_id': [f'track_{i:03d}'] * length,
            'frame': range(length),
            'x_um': np.random.randn(length).cumsum() * 0.1,
            'y_um': np.random.randn(length).cumsum() * 0.1,
            'time_s_zeroed': np.arange(length) * time_between_frames,
            'time_s': np.arange(length) * time_between_frames,
            'filename': [f'cell_{i}.tiff'] * length
        }
        data.append(pd.DataFrame(track_data))
    
    return pd.concat(data, ignore_index=True)


class TestFilterStubs:
    """Test filter_stubs function."""
    
    def test_filter_stubs_basic(self, sample_tracking_data):
        """Test basic filtering of short tracks."""
        # Filter tracks shorter than 1.0 seconds
        min_time = 1.0
        filtered_df = filter_stubs(sample_tracking_data, min_time)
        
        # Check that result is a DataFrame
        assert isinstance(filtered_df, pd.DataFrame)
        
        # Check that all remaining tracks are longer than min_time
        track_durations = filtered_df.groupby('unique_id')['time_s_zeroed'].max()
        assert all(track_durations >= min_time)
        
        # Should have fewer tracks than original
        original_tracks = sample_tracking_data['unique_id'].nunique()
        filtered_tracks = filtered_df['unique_id'].nunique()
        assert filtered_tracks <= original_tracks
    
    def test_filter_stubs_no_removal(self, sample_tracking_data):
        """Test filtering with very small min_time (no tracks removed)."""
        min_time = 0.1  # Very small, should keep all tracks
        filtered_df = filter_stubs(sample_tracking_data, min_time)
        
        # Should keep all tracks
        original_tracks = sample_tracking_data['unique_id'].nunique()
        filtered_tracks = filtered_df['unique_id'].nunique()
        assert filtered_tracks == original_tracks
    
    def test_filter_stubs_remove_all(self, sample_tracking_data):
        """Test filtering with very large min_time (all tracks removed)."""
        min_time = 100.0  # Very large, should remove all tracks
        filtered_df = filter_stubs(sample_tracking_data, min_time)
        
        # Should remove all tracks (or keep none that meet criteria)
        filtered_tracks = filtered_df['unique_id'].nunique()
        assert filtered_tracks == 0


class TestPathfixer:
    """Test pathfixer function."""
    
    @pytest.fixture
    def trajectory_with_jumps(self):
        """Create trajectory data with large jumps that should be split."""
        n_frames = 100
        data = {
            'unique_id': ['track_001'] * n_frames,
            'frame': range(n_frames),
            'x_um': np.concatenate([
                np.linspace(0, 1, 30),        # Normal segment
                np.linspace(10, 11, 30),      # Jump (large segment)
                np.linspace(11, 12, 40)       # Normal segment
            ]),
            'y_um': np.concatenate([
                np.linspace(0, 1, 30),
                np.linspace(0, 1, 30), 
                np.linspace(1, 2, 40)
            ]),
            'time_s': np.arange(n_frames) * 0.1,
            'filename': ['test_cell.tiff'] * n_frames
        }
        return pd.DataFrame(data)
    
    def test_pathfixer_basic(self, trajectory_with_jumps):
        """Test basic pathfixer functionality."""
        cleaned_df, removed_ids, report = pathfixer(
            trajectory_with_jumps,
            segment_length_threshold=5.0,  # Large jumps > 5 μm will trigger splits
            remove_short_tracks=True,
            min_track_length_seconds=1.0,
            time_between_frames=0.1
        )
        
        assert isinstance(cleaned_df, pd.DataFrame)
        assert isinstance(removed_ids, list)
        assert isinstance(report, dict)
        
        # Check report structure
        expected_keys = [
            'original_track_count', 'final_track_count',
            'tracks_split', 'tracks_removed', 'total_segments_removed'
        ]
        for key in expected_keys:
            assert key in report
    
    def test_pathfixer_no_splits_needed(self, sample_tracking_data):
        """Test pathfixer with data that doesn't need splitting."""
        cleaned_df, removed_ids, report = pathfixer(
            sample_tracking_data,
            segment_length_threshold=10.0,  # Very high threshold
            remove_short_tracks=False,
            min_track_length_seconds=0.1,
            time_between_frames=0.1
        )
        
        # Should not split any tracks
        assert report['tracks_split'] == 0
        assert report['total_segments_removed'] == 0
        
        # Track count might change due to short track removal
        assert report['final_track_count'] >= 0


def test_example_function():
    """Test the example function in the package."""
    from SPTnano import example_function
    
    result = example_function("hello", keyword_argument=" world")
    assert result == "hello world"
    
    # Test with default keyword argument
    result = example_function("test")
    assert result == "testdefault" 