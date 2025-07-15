"""Test transformer module."""

import pytest
import torch
import pandas as pd
import numpy as np
from SPTnano.transformer import (
    TransformerMotionEncoder, 
    PandasTrajectoryDataset, 
    MotionTrainer
)


@pytest.fixture
def sample_transformer_data():
    """Create sample data for transformer testing."""
    n_tracks = 5
    window_size = 60
    
    data_list = []
    for track_id in range(n_tracks):
        for window_start in range(0, 200, 30):  # Multiple windows per track
            window_data = {
                'unique_id': [f'track_{track_id:03d}'] * window_size,
                'frame': range(window_start, window_start + window_size),
                'x': np.random.randn(window_size).cumsum() * 0.1,
                'y': np.random.randn(window_size).cumsum() * 0.1,
                'time_s': np.arange(window_size) * 0.01,
                'condition': ['test_condition'] * window_size,
                'filename': [f'cell_{track_id}.tiff'] * window_size,
                'speed_um_s': np.abs(np.random.randn(window_size)),
                'direction_rad': np.random.uniform(-np.pi, np.pi, window_size),
                'instant_diff_coeff': np.abs(np.random.randn(window_size))
            }
            data_list.append(pd.DataFrame(window_data))
    
    return pd.concat(data_list, ignore_index=True)


class TestTransformerMotionEncoder:
    """Test TransformerMotionEncoder model."""
    
    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = TransformerMotionEncoder(input_dim=3, embed_dim=64, num_heads=4)
        
        assert model.input_proj.in_features == 3
        assert model.input_proj.out_features == 64
        assert model.cls_token.shape == (1, 1, 64)
    
    def test_model_forward_pass(self):
        """Test forward pass with sample data."""
        model = TransformerMotionEncoder(input_dim=3, embed_dim=64, num_heads=4)
        
        # Create sample input [batch_size, sequence_length, features]
        batch_size, seq_len, input_dim = 8, 60, 3
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = model(x)
        
        assert output.shape == (batch_size, 64)  # [batch_size, embed_dim]
        assert not torch.isnan(output).any()
    
    def test_model_with_different_sequence_lengths(self):
        """Test model handles different sequence lengths."""
        model = TransformerMotionEncoder(input_dim=3, embed_dim=32, num_heads=2)
        
        # Test with different sequence lengths
        for seq_len in [30, 60, 120]:
            x = torch.randn(4, seq_len, 3)
            output = model(x)
            assert output.shape == (4, 32)


class TestPandasTrajectoryDataset:
    """Test PandasTrajectoryDataset functionality."""
    
    def test_dataset_initialization(self, sample_transformer_data):
        """Test dataset initialization."""
        dataset = PandasTrajectoryDataset(
            sample_transformer_data, 
            window_size=60, 
            overlap=30
        )
        
        assert len(dataset) > 0
        assert hasattr(dataset, 'features')
        assert dataset.window_size == 60
        assert dataset.overlap == 30
    
    def test_dataset_getitem(self, sample_transformer_data):
        """Test dataset item retrieval."""
        dataset = PandasTrajectoryDataset(
            sample_transformer_data, 
            window_size=60, 
            overlap=30
        )
        
        if len(dataset) > 0:
            item = dataset[0]
            
            assert isinstance(item, torch.Tensor)
            assert item.shape[0] == 60  # window_size
            assert item.shape[1] == 3   # features (speed, direction, diff_coeff)
            assert not torch.isnan(item).any()
    
    def test_track_info_extraction(self, sample_transformer_data):
        """Test track information extraction."""
        dataset = PandasTrajectoryDataset(
            sample_transformer_data, 
            window_size=60, 
            overlap=30
        )
        
        track_info = dataset.get_track_info()
        
        assert isinstance(track_info, pd.DataFrame)
        assert 'unique_id' in track_info.columns
        assert 'window_index' in track_info.columns
        assert len(track_info) == len(dataset)


class TestMotionTrainer:
    """Test MotionTrainer functionality."""
    
    @pytest.fixture
    def simple_trainer(self, sample_transformer_data):
        """Create a simple trainer for testing."""
        from torch.utils.data import DataLoader
        
        dataset = PandasTrajectoryDataset(
            sample_transformer_data, 
            window_size=60, 
            overlap=30
        )
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        model = TransformerMotionEncoder(input_dim=3, embed_dim=32, num_heads=2)
        
        trainer = MotionTrainer(
            model, 
            dataloader, 
            val_loader=None,
            device='cpu',
            use_tensorboard=False
        )
        
        return trainer
    
    def test_trainer_initialization(self, simple_trainer):
        """Test trainer initialization."""
        assert simple_trainer.model is not None
        assert simple_trainer.train_loader is not None
        assert simple_trainer.device == 'cpu'
        assert hasattr(simple_trainer, 'optimizer')
    
    def test_single_training_step(self, simple_trainer):
        """Test a single training step."""
        initial_loss = None
        
        # Run one training step
        for batch in simple_trainer.train_loader:
            loss = simple_trainer._train_step(batch)
            initial_loss = loss
            break  # Only test one batch
        
        assert initial_loss is not None
        assert isinstance(initial_loss, (int, float))
        assert initial_loss >= 0
    
    def test_embedding_extraction(self, simple_trainer):
        """Test embedding extraction."""
        embeddings = simple_trainer.extract_embeddings(simple_trainer.train_loader)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[1] == 32  # embed_dim
        assert len(embeddings) > 0
    
    def test_clustering(self, simple_trainer):
        """Test clustering functionality."""
        embeddings = simple_trainer.extract_embeddings(simple_trainer.train_loader)
        
        if len(embeddings) >= 3:  # Need at least 3 points for clustering
            clusters = simple_trainer.cluster_embeddings(embeddings, n_clusters=2)
            
            assert len(clusters) == len(embeddings)
            assert len(np.unique(clusters)) <= 2
            assert all(cluster >= 0 for cluster in clusters)


def test_train_motion_transformer_basic(sample_transformer_data):
    """Test basic train_motion_transformer function."""
    from SPTnano.transformer import train_motion_transformer
    
    # Use minimal parameters for quick test
    trainer, datasets, split_info = train_motion_transformer(
        sample_transformer_data,
        window_size=30,
        overlap=15,
        batch_size=4,
        epochs=1,  # Just one epoch for testing
        val_split=0.2,
        test_split=0.2,
        device='cpu',
        use_tensorboard=False
    )
    
    assert trainer is not None
    assert 'train' in datasets
    assert 'test' in datasets
    assert isinstance(split_info, dict) 