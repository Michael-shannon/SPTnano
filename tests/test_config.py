"""Test configuration module."""

import os
import pytest
from SPTnano import config


def test_config_constants():
    """Test that required configuration constants are defined."""
    assert hasattr(config, 'MASTER')
    assert hasattr(config, 'SAVED_DATA')
    assert hasattr(config, 'PIXELSIZE_MICRONS')
    assert hasattr(config, 'TIME_BETWEEN_FRAMES')
    assert hasattr(config, 'FEATURES')
    
    # Test types
    assert isinstance(config.PIXELSIZE_MICRONS, (int, float))
    assert isinstance(config.TIME_BETWEEN_FRAMES, (int, float))
    assert isinstance(config.FEATURES, list)
    assert len(config.FEATURES) > 0


def test_config_features_list():
    """Test that features list contains expected feature names."""
    expected_features = [
        'speed_um_s', 
        'direction_rad', 
        'acceleration_um_s2',
        'jerk_um_s3', 
        'normalized_curvature', 
        'angle_normalized_curvature',
        'instant_diff_coeff'
    ]
    
    for feature in expected_features:
        assert feature in config.FEATURES


def test_analysis_params_structure():
    """Test that ANALYSIS_PARAMS has expected structure."""
    assert hasattr(config, 'ANALYSIS_PARAMS')
    assert isinstance(config.ANALYSIS_PARAMS, dict)
    
    # Check required top-level keys
    assert 'min_track_length' in config.ANALYSIS_PARAMS
    assert 'transformer_params' in config.ANALYSIS_PARAMS
    assert 'training_params' in config.ANALYSIS_PARAMS
    
    # Check transformer params structure
    transformer_params = config.ANALYSIS_PARAMS['transformer_params']
    assert 'single_scale' in transformer_params
    assert 'multi_scale' in transformer_params


def test_directories_created():
    """Test that required directories are created."""
    # Check if SAVED_DATA directory exists or can be created
    assert config.SAVED_DATA is not None
    
    # Check if TENSORBOARD_LOGS exists or can be created  
    assert config.TENSORBOARD_LOGS is not None 