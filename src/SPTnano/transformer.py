import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

class TransformerMotionEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64, num_heads=4, ff_dim=128, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x: [B, T, 3]
        B, T, _ = x.shape
        x = self.input_proj(x)  # [B, T, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, T+1, embed_dim]
        x = self.transformer_encoder(x.permute(1, 0, 2))  # [T+1, B, embed_dim]
        x = x[0]  # take CLS token [B, embed_dim]
        return self.norm(x)

class TimeWindowDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)  # [N, 60, 3]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class PandasTrajectoryDataset(Dataset):
    """
    Dataset class for extracting time windows from pandas trajectory dataframe.
    Creates sequences of (dx, dy, direction) motion features.
    """
    
    def __init__(self, instant_df, window_size=60, overlap=30, min_track_length=60):
        """
        Args:
            instant_df: DataFrame with trajectory data
            window_size: Number of frames per window
            overlap: Number of overlapping frames between windows
            min_track_length: Minimum track length to consider
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = window_size - overlap
        self.min_track_length = min_track_length
        
        # Sort by unique_id and frame to ensure temporal order
        self.df = instant_df.sort_values(['unique_id', 'frame']).reset_index(drop=True)
        
        # Generate windows
        self.windows = self._generate_windows()
        
    def _generate_windows(self):
        """Generate all valid windows from the trajectory data"""
        windows = []
        
        # Group by track ID
        for unique_id, track_data in self.df.groupby('unique_id'):
            track_data = track_data.sort_values('frame').reset_index(drop=True)
            
            # Skip short tracks
            if len(track_data) < self.min_track_length:
                continue
                
            # Generate overlapping windows for this track
            time_window_num = 0
            for start_idx in range(0, len(track_data) - self.window_size + 1, self.step_size):
                end_idx = start_idx + self.window_size
                window_data = track_data.iloc[start_idx:end_idx].copy()
                
                # Extract motion features for this window
                features = self._extract_features(window_data)
                
                if features is not None:
                    # Create window_uid that matches features.py format: unique_id_timewindow_framestart_frameend
                    frame_start = window_data.iloc[0]['frame']
                    frame_end = window_data.iloc[-1]['frame']
                    window_uid = f"{unique_id}_{time_window_num}_{frame_start}_{frame_end}"
                    
                    windows.append({
                        'features': features,
                        'unique_id': unique_id,
                        'start_frame': frame_start,
                        'end_frame': frame_end,
                        'window_idx': len(windows),
                        'window_uid': window_uid,  # NEW: Add window_uid for mapping
                        'time_window': time_window_num,  # NEW: Add time window number
                        'condition': track_data['condition'].iloc[0]  # Store condition for analysis
                    })
                    time_window_num += 1
                    
        return windows
    
    def _extract_features(self, window_data):
        """
        Extract motion features from a window of trajectory data.
        Returns array of shape (window_size, 3) with features: [dx, dy, direction]
        """
        try:
            # Calculate dx, dy (displacement between consecutive points)
            dx = np.diff(window_data['x_um'].values, prepend=window_data['x_um'].iloc[0])
            dy = np.diff(window_data['y_um'].values, prepend=window_data['y_um'].iloc[0])
            
            # Calculate direction
            if 'direction_rad' in window_data.columns:
                direction = window_data['direction_rad'].values
            else:
                direction = np.arctan2(dy, dx)
                direction = np.nan_to_num(direction, 0)
            
            # Stack features: [dx, dy, direction] - only 3 features as requested
            features = np.column_stack([dx, dy, direction])
            
            # Handle any remaining NaN or infinite values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        return {
            'features': torch.FloatTensor(window['features']),
            'unique_id': window['unique_id'],
            'start_frame': window['start_frame'],
            'end_frame': window['end_frame'],
            'window_idx': window['window_idx'],
            'window_uid': window['window_uid'],
            'time_window': window['time_window'],
            'condition': window['condition']
        }
    
    def get_labels(self):
        """Return condition labels for each window (for supervised learning if needed)"""
        return [window['condition'] for window in self.windows]
    
    def get_track_info(self):
        """Return track information for mapping back to original data"""
        return [{
            'unique_id': window['unique_id'],
            'start_frame': window['start_frame'],
            'end_frame': window['end_frame'],
            'window_uid': window['window_uid'],  # NEW: Include window_uid for mapping
            'time_window': window['time_window'],  # NEW: Include time window number
            'condition': window['condition']
        } for window in self.windows]

def contrastive_loss(z_i, z_j, temperature=0.5):
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    N = z_i.shape[0]
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0)
    labels = labels.to(z_i.device)

    mask = ~torch.eye(2 * N, dtype=bool).to(z_i.device)
    logits = similarity_matrix[mask].view(2 * N, -1)
    positives = torch.exp(similarity_matrix[torch.arange(2 * N), labels] / temperature)
    denominator = torch.sum(torch.exp(logits / temperature), dim=1)
    loss = -torch.log(positives / denominator)
    return loss.mean()

class MotionTrainer:
    def __init__(self, model, dataloader, lr=1e-4, device='cuda', use_tensorboard=False, tensorboard_log_dir=None, augmentation_strategy='basic', use_scheduler=False):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        self.losses = []
        
        # Learning rate scheduler
        self.use_scheduler = use_scheduler
        self.scheduler = None
        if use_scheduler:
            # Use ReduceLROnPlateau scheduler - reduces LR when loss plateaus
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
            
            # 🔥 Alternative scheduler options (uncomment to try):
            # Option 1: More conservative plateau detection
            # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     self.optimizer, mode='min', factor=0.8, patience=8, verbose=True
            # )
            
            # Option 2: Step-based reduction (every N epochs)
            # self.scheduler = torch.optim.lr_scheduler.StepLR(
            #     self.optimizer, step_size=10, gamma=0.8
            # )
            
            # Option 3: Cosine annealing (smooth reduction)
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     self.optimizer, T_max=epochs, eta_min=1e-6
            # )
        
        # TensorBoard setup
        self.use_tensorboard = use_tensorboard
        self.writer = None
        if use_tensorboard and tensorboard_log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
                print(f"   📊 TensorBoard logging enabled: {tensorboard_log_dir}")
            except ImportError:
                print("   ⚠ TensorBoard not available. Install with: mamba install tensorboard")
                self.use_tensorboard = False
        
        # Augmentation setup
        self.augmentation_strategy = augmentation_strategy
        self._setup_augmentation()
    
    def _setup_augmentation(self):
        """Setup augmentation function based on strategy"""
        try:
            from .augmentations import get_augmentation_function
            base_augmentation_fn = get_augmentation_function(self.augmentation_strategy)
            # Wrap the augmentation function to ensure device consistency
            def device_aware_augmentation(x):
                aug_x = base_augmentation_fn(x)
                if isinstance(aug_x, torch.Tensor):
                    return aug_x.to(self.device)
                return torch.tensor(aug_x, device=self.device)
            self.augmentation_fn = device_aware_augmentation
            print(f"   🔄 Augmentation strategy: {self.augmentation_strategy}")
        except ImportError:
            print("   ⚠ Augmentation module not available, using basic noise")
            def basic_noise_augmentation(x):
                return x + 0.01 * torch.randn_like(x, device=self.device)
            self.augmentation_fn = basic_noise_augmentation
    
    def train(self, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in self.dataloader:
                # Handle both old format (tensor) and new format (dict)
                if isinstance(batch, dict):
                    batch_features = batch['features'].to(self.device)
                else:
                    batch_features = batch.to(self.device)
                
                x1 = batch_features
                x2 = self.augmentation_fn(batch_features)  # Now guaranteed to be on the correct device
                z1 = self.model(x1)
                z2 = self.model(x2)
                loss = contrastive_loss(z1, z2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.dataloader)
            self.losses.append(avg_loss)
            
            # Update learning rate scheduler
            if self.use_scheduler and self.scheduler:
                self.scheduler.step(avg_loss)
            
            # Log to TensorBoard
            if self.use_tensorboard and self.writer:
                self.writer.add_scalar('Loss/Train', avg_loss, epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
                # Also log learning rate changes as events
                if epoch > 0 and len(self.losses) > 1:
                    prev_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('LR_vs_Loss', prev_lr, avg_loss)
                self.writer.flush()  # Ensure immediate logging
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, LR = {current_lr:.2e}")
        
        # Close TensorBoard writer
        if self.use_tensorboard and self.writer:
            self.writer.close()

    def plot_loss_curve(self):
        """Plot the training loss curve"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    def extract_embeddings(self, dataloader=None):
        """Extract embeddings from the dataloader"""
        if dataloader is None:
            dataloader = self.dataloader
            
        self.model.eval()
        all_embeddings = []
        track_info = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle both old format (tensor) and new format (dict)
                if isinstance(batch, dict):
                    batch_features = batch['features'].to(self.device)
                    # Store track info if available
                    if hasattr(dataloader.dataset, 'get_track_info'):
                        batch_size = batch_features.shape[0]
                        batch_indices = range(len(track_info), len(track_info) + batch_size)
                        # This is approximate - for exact mapping you'd need batch indices
                else:
                    batch_features = batch.to(self.device)
                
                embeddings = self.model(batch_features)
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)

    def cluster_embeddings(self, embeddings, n_clusters=5):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)

    def save_model(self, path):
        """Save model state dict and training info"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'augmentation_strategy': self.augmentation_strategy,
            'device': str(self.device)
        }
        torch.save(save_dict, path)
        print(f"✅ Model saved to: {path}")

    def load_model(self, path, load_optimizer=True):
        """Load model state dict and optionally optimizer state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Load optimizer if requested and available
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history
        if 'losses' in checkpoint:
            self.losses = checkpoint['losses']
        
        # Load augmentation strategy
        if 'augmentation_strategy' in checkpoint:
            self.augmentation_strategy = checkpoint['augmentation_strategy']
            self._setup_augmentation()
        
        print(f"✅ Model loaded from: {path}")
        print(f"   Previous training epochs: {len(self.losses)}")
        if 'augmentation_strategy' in checkpoint:
            print(f"   Augmentation strategy: {checkpoint['augmentation_strategy']}")
    
    def resume_training(self, epochs=10):
        """Resume training from current state"""
        print(f"🔄 Resuming training from epoch {len(self.losses) + 1}")
        self.train(epochs)

# Convenience functions for easy use from notebooks
def create_trajectory_dataset(instant_df, window_size=60, overlap=30, min_track_length=60):
    """Create a trajectory dataset from pandas DataFrame"""
    return PandasTrajectoryDataset(instant_df, window_size, overlap, min_track_length)

def train_motion_transformer(instant_df, 
                           window_size=60, 
                           overlap=30, 
                           batch_size=64, 
                           epochs=10, 
                           test_split=0.2,
                           device='auto',
                           use_tensorboard=False,
                           tensorboard_log_dir=None,
                           augmentation_strategy='basic',
                           save_model_path=None,
                           use_scheduler=False):
    """
    Complete pipeline to train a motion transformer on trajectory data
    
    Args:
        instant_df: DataFrame with trajectory data
        window_size: Number of frames per window
        overlap: Overlap between windows
        batch_size: Training batch size
        epochs: Number of training epochs
        test_split: Fraction of data for testing
        device: 'auto', 'cuda', or 'cpu'
        use_tensorboard: Enable TensorBoard logging
        tensorboard_log_dir: Directory for TensorBoard logs
        augmentation_strategy: Augmentation strategy ('basic', 'comprehensive', etc.)
        save_model_path: Path to save trained model (optional)
    
    Returns:
        trainer: Trained MotionTrainer object
        train_dataset: Training dataset
        test_dataset: Test dataset
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Split data by tracks to avoid leakage
    unique_tracks = instant_df['unique_id'].unique()
    np.random.seed(42)
    train_tracks = np.random.choice(unique_tracks, size=int((1-test_split) * len(unique_tracks)), replace=False)
    test_tracks = np.setdiff1d(unique_tracks, train_tracks)
    
    train_df = instant_df[instant_df['unique_id'].isin(train_tracks)]
    test_df = instant_df[instant_df['unique_id'].isin(test_tracks)]
    
    print(f"Train tracks: {len(train_tracks)}, Test tracks: {len(test_tracks)}")
    
    # Create datasets
    train_dataset = PandasTrajectoryDataset(train_df, window_size, overlap)
    test_dataset = PandasTrajectoryDataset(test_df, window_size, overlap)
    
    print(f"Train windows: {len(train_dataset)}, Test windows: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create and train model
    model = TransformerMotionEncoder(input_dim=3)  # 3 features: dx, dy, direction
    trainer = MotionTrainer(model, train_loader, device=device, 
                           use_tensorboard=use_tensorboard, 
                           tensorboard_log_dir=tensorboard_log_dir,
                           augmentation_strategy=augmentation_strategy,
                           use_scheduler=use_scheduler)
    
    print("Starting training...")
    trainer.train(epochs=epochs)
    
    # Save model if path provided
    if save_model_path:
        trainer.save_model(save_model_path)
    
    return trainer, train_dataset, test_dataset

def train_multi_scale_transformers(
    df, 
    scales=None,
    batch_size=64, 
    epochs=10, 
    test_split=0.2,
    device='auto',
    n_clusters=5,
    use_tensorboard=False,
    augmentation_strategy='basic',
    save_models=True,
    use_scheduler=False,
    single_scale_mode=False  # NEW: Add option to use only 60f scale with direct mapping
):
    """
    Train transformers across multiple temporal scales or single scale.
    
    Parameters:
    -----------
    single_scale_mode : bool
        If True, only train 60f scale and use direct window_uid mapping.
        If False, use full multi-scale approach.
    """
    
    if single_scale_mode:
        print("🎯 SINGLE-SCALE MODE: Training only 60f scale with direct window_uid mapping")
        # Override scales to only use 60f
        scales = [{'window_size': 60, 'overlap': 30}]
    
    # Default scales if none provided
    if scales is None:
        scales = [
            {'window_size': 30, 'overlap': 15},
            {'window_size': 60, 'overlap': 30}, 
            {'window_size': 120, 'overlap': 60},
            {'window_size': 240, 'overlap': 120}
        ]
    
    import pandas as pd
    import numpy as np
    from torch.utils.data import DataLoader
    import os
    from datetime import datetime
    
    print("=" * 60)
    print("MULTI-SCALE TRANSFORMER TRAINING")
    print("=" * 60)
    
    # Setup TensorBoard logging directories
    tensorboard_base_dir = None
    if use_tensorboard:
        try:
            # Import config from parent directory
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            import config
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tensorboard_base_dir = os.path.join(config.TENSORBOARD_LOGS, f"multi_scale_{timestamp}")
            os.makedirs(tensorboard_base_dir, exist_ok=True)
            print(f"📊 TensorBoard logs will be saved to: {tensorboard_base_dir}")
        except Exception as e:
            print(f"⚠ TensorBoard setup failed: {e}")
            use_tensorboard = False
    
    scale_results = {}
    
    # Train transformer for each scale
    for i, scale_config in enumerate(scales):
        window_size = scale_config['window_size']
        overlap = scale_config['overlap']
        
        print(f"\n🔬 SCALE {i+1}: {window_size}-frame windows ({overlap}-frame overlap)")
        print(f"   Step size: {window_size - overlap} frames")
        
        # Check if any tracks are long enough for this scale
        max_track_length = df.groupby('unique_id')['frame'].count().max()
        if max_track_length < window_size:
            print(f"   ⚠ Skipping scale: No tracks long enough ({max_track_length} < {window_size})")
            continue
            
        # Setup scale-specific TensorBoard logging
        scale_tensorboard_dir = None
        if use_tensorboard and tensorboard_base_dir:
            scale_tensorboard_dir = os.path.join(tensorboard_base_dir, f"scale_{window_size}f")
            os.makedirs(scale_tensorboard_dir, exist_ok=True)
        
        # Train transformer for this scale
        try:
            # Setup model save path
            scale_model_path = None
            if save_models and tensorboard_base_dir:
                scale_model_path = os.path.join(tensorboard_base_dir, f"model_{window_size}f.pt")
            
            trainer, train_dataset, test_dataset = train_motion_transformer(
                df,
                window_size=window_size,
                overlap=overlap,
                batch_size=batch_size,
                epochs=epochs,
                test_split=test_split,
                device=device,
                use_tensorboard=use_tensorboard,
                tensorboard_log_dir=scale_tensorboard_dir,
                augmentation_strategy=augmentation_strategy,
                save_model_path=scale_model_path,
                use_scheduler=use_scheduler
            )
            
            print(f"   ✓ Training completed: {len(train_dataset)} train, {len(test_dataset)} test windows")
            
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            continue
            
        # Extract embeddings and cluster
        print(f"   📊 Extracting embeddings and clustering...")
        
        all_dataset = PandasTrajectoryDataset(df, window_size=window_size, overlap=overlap)
        all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)
        
        embeddings = trainer.extract_embeddings(all_loader)
        cluster_labels = trainer.cluster_embeddings(embeddings, n_clusters=n_clusters)
        
        # Get metadata
        condition_labels = all_dataset.get_labels()
        track_info = all_dataset.get_track_info()
        
        print(f"   ✓ Generated {len(embeddings)} embeddings, {n_clusters} clusters")
        
        # Create scale-specific windowed dataframe
        step_size = window_size - overlap
        scale_windowed_df = pd.DataFrame({
            'time_window': [info['start_frame'] // step_size for info in track_info],
            'unique_id': [info['unique_id'] for info in track_info],
            'cluster': cluster_labels,
            'condition': condition_labels,
            'start_frame': [info['start_frame'] for info in track_info],
            'end_frame': [info['end_frame'] for info in track_info],
            'window_uid': [info['window_uid'] for info in track_info],  # NEW: Include window_uid for mapping
            'window_size': window_size,
            'scale_id': i
        })
        
        # Store results for this scale
        scale_results[f'scale_{window_size}f'] = {
            'trainer': trainer,
            'embeddings': embeddings,
            'cluster_labels': cluster_labels,
            'windowed_df': scale_windowed_df,
            'track_info': track_info,
            'window_size': window_size,
            'overlap': overlap,
            'n_windows': len(embeddings)
        }
        
        print(f"   ✓ Scale {window_size}f completed: {len(scale_windowed_df)} windows")
    
    print(f"\n🔗 INTEGRATING MULTI-SCALE RESULTS...")
    
    if single_scale_mode:
        # Simple direct mapping for single scale
        print("   🎯 SINGLE-SCALE MODE: Using direct window_uid mapping...")
        scale_name = list(scale_results.keys())[0]  # Should be 'scale_60f'
        scale_info = scale_results[scale_name]
        
        integrated_instant_df = map_single_scale_direct(
            df.copy(),
            scale_info['windowed_df'],
            scale_info['cluster_labels']
        )
        
        # For single-scale mode, set cluster_scale column
        integrated_instant_df['cluster_scale'] = integrated_instant_df['final_cluster'].apply(
            lambda x: '60f' if x != -1 else 'none'
        )
        
        # Create signature columns for single scale
        signature_cols = ['cluster_scale_60f']
        integrated_instant_df['multi_scale_signature'] = integrated_instant_df['cluster_scale_60f'].astype(str)
        
    else:
        # 🔥 NEW: Use mixed mapping approach for multi-scale
        integrated_instant_df = map_multiscale_clusters_to_instant(
            instant_df=df,
            scale_results=scale_results,
            track_col='unique_id',
            frame_col='frame'
        )
        
        # 🔥 NEW: Multi-scale cluster assignment (not just shortest priority)
        print("   📊 Creating multi-scale cluster signatures...")
        
        # Option 1: Shortest scale priority (backward compatibility)
        scale_names_by_size = sorted(scale_results.keys(), 
                                    key=lambda x: scale_results[x]['window_size'])
        
        integrated_instant_df['final_cluster'] = np.nan
        integrated_instant_df['cluster_scale'] = 'none'
        
        for i, row in integrated_instant_df.iterrows():
            for scale_name in scale_names_by_size:
                scale_suffix = scale_name.replace('scale_', '')
                cluster_val = row[f'cluster_scale_{scale_suffix}']
                if cluster_val != -1:  # Changed from pd.notna to != -1
                    integrated_instant_df.at[i, 'final_cluster'] = cluster_val
                    integrated_instant_df.at[i, 'cluster_scale'] = scale_suffix
                    break
        
        # Create signature columns for multi-scale
        signature_cols = [f'cluster_scale_{scale_name.replace("scale_", "")}' for scale_name in scale_results.keys()]
        integrated_instant_df['multi_scale_signature'] = integrated_instant_df[signature_cols].apply(
            lambda row: '_'.join(row.astype(str)), axis=1
        )
    
    # Report final mapping results
    frame_assignments = (~integrated_instant_df['final_cluster'].isna()).sum()
    print(f"   ✅ Mapped clusters to {frame_assignments:,} trajectory points")
    
    # Create summary statistics  
    scale_coverage = {}
    expected_coverage = {}
    for scale_name in scale_results.keys():
        # Extract scale suffix (e.g., '60f' from 'scale_60f')
        scale_suffix = scale_name.replace('scale_', '')
        
        # Count actual valid assignments (not -1)
        coverage = (integrated_instant_df[f'cluster_scale_{scale_suffix}'] != -1).sum()
        scale_coverage[scale_name] = coverage
        
        # 🔍 DEBUGGING: Calculate expected coverage based on track lengths
        window_size = scale_results[scale_name]['window_size']
        tracks = integrated_instant_df.groupby('unique_id').size()
        expected_points = tracks[tracks >= window_size].sum()
        expected_coverage[scale_name] = expected_points
    
    print(f"\n📊 COVERAGE SUMMARY (Valid Assignments Only):")
    print(f"{'Scale':<10} {'Actual':<10} {'Expected':<10} {'Efficiency':<10}")
    print("-" * 45)
    for scale_name in scale_results.keys():
        actual = scale_coverage[scale_name]
        expected = expected_coverage[scale_name]
        efficiency = (actual / expected * 100) if expected > 0 else 0
        actual_pct = actual / len(integrated_instant_df) * 100
        expected_pct = expected / len(integrated_instant_df) * 100
        print(f"{scale_name:<10} {actual:>6,} ({actual_pct:4.1f}%) {expected:>6,} ({expected_pct:4.1f}%) {efficiency:>6.1f}%")
    
    final_coverage = (~integrated_instant_df['final_cluster'].isna()).sum()
    final_percentage = final_coverage / len(integrated_instant_df) * 100
    print(f"\n   Final assignment: {final_coverage:,} points ({final_percentage:.1f}%)")
    
    # 🔍 DEBUGGING: Show mapping efficiency issues
    print(f"\n🔍 MAPPING EFFICIENCY ANALYSIS:")
    total_expected = sum(expected_coverage.values())
    total_actual = sum(scale_coverage.values())
    if total_expected > 0:
        mapping_efficiency = total_actual / total_expected * 100
        print(f"   Overall mapping efficiency: {mapping_efficiency:.1f}%")
        if mapping_efficiency < 50:
            print(f"   ⚠️ Low mapping efficiency suggests coordinate mismatch issues")
            print(f"   💡 This explains why coverage is similar across scales")
    
    print(f"\n📊 WINDOW CREATION vs FRAME MAPPING:")
    for scale_name, scale_info in scale_results.items():
        n_windows = scale_info['n_windows'] 
        mapped_points = scale_coverage[scale_name]
        window_size = scale_info['window_size']
        theoretical_points = n_windows * window_size
        print(f"   {scale_name}: {n_windows:,} windows → {theoretical_points:,} theoretical points → {mapped_points:,} actual points")
    
    # 🔥 NEW: Multi-scale analysis summary
    print(f"\n📊 MULTI-SCALE BEHAVIORAL ANALYSIS:")
    total_frames = len(integrated_instant_df)
    
    # Count frames with different levels of scale coverage
    no_coverage = (integrated_instant_df[signature_cols] == -1).all(axis=1).sum()
    partial_coverage = ((integrated_instant_df[signature_cols] == -1).any(axis=1) & 
                       ~(integrated_instant_df[signature_cols] == -1).all(axis=1)).sum()
    full_coverage = (~(integrated_instant_df[signature_cols] == -1).any(axis=1)).sum()
    
    print(f"   🔍 No scale coverage: {no_coverage:,} points ({no_coverage/total_frames*100:.1f}%)")
    print(f"   📊 Partial coverage: {partial_coverage:,} points ({partial_coverage/total_frames*100:.1f}%)")  
    print(f"   🎯 Full multi-scale: {full_coverage:,} points ({full_coverage/total_frames*100:.1f}%)")
    
    # Sample of multi-scale signatures
    valid_signatures = integrated_instant_df[integrated_instant_df['final_cluster'].notna()]['multi_scale_signature'].value_counts().head(5)
    print(f"\n🧬 Top 5 behavioral signatures:")
    for signature, count in valid_signatures.items():
        print(f"   {signature}: {count:,} frames")
    
    # Combine all windowed dataframes
    all_windowed_dfs = []
    for scale_name, scale_info in scale_results.items():
        windowed_df = scale_info['windowed_df'].copy()
        windowed_df['scale_name'] = scale_name
        all_windowed_dfs.append(windowed_df)
    
    combined_windowed_df = pd.concat(all_windowed_dfs, ignore_index=True) if all_windowed_dfs else pd.DataFrame()
    
    # Final results package
    results = {
        'integrated_instant_df': integrated_instant_df,
        'combined_windowed_df': combined_windowed_df,
        'scale_results': scale_results,
        'scale_coverage': scale_coverage,
        'scales_config': scales,
        'parameters': {
            'n_clusters': n_clusters,
            'batch_size': batch_size,
            'epochs': epochs,
            'test_split': test_split
        }
    }
    
    print(f"\n✅ MULTI-SCALE ANALYSIS COMPLETED!")
    print(f"   📊 Scales trained: {len(scale_results)}")
    print(f"   🎯 Total windows: {sum(s['n_windows'] for s in scale_results.values()):,}")
    print(f"   📍 Frame coverage: {final_percentage:.1f}%")
    
    return results

def map_single_scale_direct(
    instant_df: pd.DataFrame,
    windowed_df: pd.DataFrame,
    cluster_labels: np.ndarray
) -> pd.DataFrame:
    """
    Simple direct mapping using window_uid for single-scale mode.
    This bypasses all the complex multi-scale logic.
    """
    print("   🎯 Using direct window_uid mapping for single scale...")
    
    # Always initialize the final_cluster column
    result_df = instant_df.copy()
    result_df['final_cluster'] = -1  # Initialize with -1 (unassigned)
    result_df['cluster_scale_60f'] = -1  # Initialize scale-specific column
    
    # Check for required columns
    if 'window_uid' not in instant_df.columns:
        print(f"   ❌ ERROR: No window_uid column in instant_df")
        print(f"   💡 Make sure your data was processed with the updated features.py")
        return result_df
        
    # Create windowed dataframe with cluster assignments
    # The windowed_df from transformer should be a list of track info dicts
    if isinstance(windowed_df, list):
        # Convert list of track info to DataFrame
        windowed_data = []
        for i, window_info in enumerate(windowed_df):
            if i < len(cluster_labels):
                windowed_data.append({
                    'window_uid': window_info['window_uid'],
                    'unique_id': window_info['unique_id'],
                    'start_frame': window_info['start_frame'],
                    'end_frame': window_info['end_frame'],
                    'time_window': window_info['time_window'],
                    'cluster_scale_60f': cluster_labels[i]
                })
        windowed_with_clusters = pd.DataFrame(windowed_data)
    else:
        # Assume it's already a DataFrame
        windowed_with_clusters = windowed_df.copy()
        if 'cluster_scale_60f' not in windowed_with_clusters.columns:
            windowed_with_clusters['cluster_scale_60f'] = cluster_labels
    
    if 'window_uid' not in windowed_with_clusters.columns:
        print(f"   ❌ ERROR: No window_uid column in windowed_df")
        print(f"   💡 The transformer windowed data doesn't have window_uid")
        return result_df
        
    # Create mapping dictionary from windowed data
    window_cluster_map = dict(zip(
        windowed_with_clusters['window_uid'], 
        windowed_with_clusters['cluster_scale_60f']
    ))
    
    print(f"   📊 Created mapping for {len(window_cluster_map)} windows")
    
    # Map clusters to instant dataframe using window_uid
    mapped_count = 0
    for idx, row in result_df.iterrows():
        window_uid = row['window_uid']
        if pd.notna(window_uid) and window_uid in window_cluster_map:
            cluster_id = window_cluster_map[window_uid]
            result_df.at[idx, 'cluster_scale_60f'] = cluster_id
            result_df.at[idx, 'final_cluster'] = cluster_id
            mapped_count += 1
    
    print(f"   ✅ Successfully mapped {mapped_count:,} frames using window_uid")
    coverage = (mapped_count / len(result_df)) * 100
    print(f"   📊 Coverage: {coverage:.1f}%")
    
    return result_df

def map_clusters_to_instant_relative_to_60f(
    instant_df: pd.DataFrame,
    windowed_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    scale_name: str
) -> pd.DataFrame:
    """
    Map windowed clusters back to instant dataframe using a relative approach:
    - For 60f scale: Use direct window_uid mapping (matches original windowing)
    - For other scales: Map relative to 60f scale based on frame overlap
    - If mapping fails: Assign NaN instead of using fallback methods
    
    Parameters:
    -----------
    instant_df : pd.DataFrame
        The instant trajectory dataframe
    windowed_df : pd.DataFrame
        The windowed dataframe for this specific scale
    cluster_labels : np.ndarray
        Cluster assignments for each window
    scale_name : str
        Name of the scale (e.g., '30f', '60f', '120f', '240f')
    
    Returns:
    --------
    pd.DataFrame
        Updated instant dataframe with cluster assignments
    """
    print(f"   🔧 {scale_name}: Using relative mapping approach...")
    
    # Add cluster assignments to windowed_df
    windowed_with_clusters = windowed_df.copy()
    windowed_with_clusters[f'cluster_scale_{scale_name}'] = cluster_labels
    
    # Initialize the cluster column in instant_df
    cluster_col = f'cluster_scale_{scale_name}'
    instant_df[cluster_col] = -1  # Default to -1 (unassigned)
    
    if scale_name == 'scale_60f':
        # Direct window_uid mapping for 60f scale
        print(f"   ✅ {scale_name}: Using direct window_uid mapping (matches original windowing)...")
        
        if 'window_uid' not in instant_df.columns:
            print(f"   ⚠️ No window_uid column in instant_df for {scale_name}")
            return instant_df
        if 'window_uid' not in windowed_with_clusters.columns:
            print(f"   ⚠️ No window_uid column in windowed_df for {scale_name}")
            return instant_df
            
        # Create mapping dictionary from windowed data
        window_cluster_map = dict(zip(
            windowed_with_clusters['window_uid'], 
            windowed_with_clusters[cluster_col]
        ))
        
        # Map clusters to instant dataframe using window_uid
        instant_df[cluster_col] = instant_df['window_uid'].map(window_cluster_map).fillna(-1).astype(int)
        
        # Report mapping quality
        mapped_count = (instant_df[cluster_col] != -1).sum()
        total_count = len(instant_df)
        coverage = (mapped_count / total_count) * 100
        print(f"   ✅ {scale_name}: Direct window_uid mapping - {mapped_count:,}/{total_count:,} points ({coverage:.1f}% coverage)")
        
    else:
        # For other scales, map relative to 60f scale based on frame overlap
        print(f"   🔄 {scale_name}: Mapping relative to 60f scale...")
        
        # Check if 60f clusters exist
        if 'cluster_scale_60f' not in instant_df.columns:
            print(f"   ⚠️ No 60f clusters available for relative mapping of {scale_name}")
            return instant_df
        
        # Extract window size from scale name
        try:
            window_size = int(scale_name.replace('f', ''))
        except ValueError:
            print(f"   ⚠️ Could not extract window size from {scale_name}")
            return instant_df
        
        # Map based on frame overlap with windowed data
        mapped_count = 0
        for _, window_row in windowed_with_clusters.iterrows():
            cluster_id = window_row[cluster_col]
            
            # Try to find overlapping frames in instant_df
            # Use track and approximate frame range matching
            track_id = window_row['unique_id']
            
            # Get the track data from instant_df
            track_mask = instant_df['unique_id'] == track_id
            track_data = instant_df[track_mask].copy()
            
            if len(track_data) == 0:
                continue
                
            # For this scale's window, try to find the best matching frames
            # Based on the window center and size
            try:
                # Estimate window center frame from windowed data
                # This is approximate since we don't have exact frame mapping
                track_frames = track_data['frame'].values
                n_track_frames = len(track_frames)
                
                # Estimate which frames this window might correspond to
                # This is a heuristic based on window size and overlap
                if window_size >= n_track_frames:
                    # Window covers entire track
                    frame_indices = track_data.index
                else:
                    # Try to estimate frame range
                    # This is approximate - in practice you might need more sophisticated logic
                    window_start_idx = max(0, (len(track_frames) - window_size) // 2)
                    window_end_idx = min(len(track_frames), window_start_idx + window_size)
                    frame_indices = track_data.iloc[window_start_idx:window_end_idx].index
                
                # Assign cluster to these frames
                instant_df.loc[frame_indices, cluster_col] = cluster_id
                mapped_count += len(frame_indices)
                
            except Exception as e:
                # If mapping fails for this window, skip it
                continue
        
        # Report mapping quality
        total_count = len(instant_df)
        coverage = (mapped_count / total_count) * 100
        print(f"   📊 {scale_name}: Relative mapping - {mapped_count:,}/{total_count:,} points ({coverage:.1f}% coverage)")
    
    return instant_df


def map_multiscale_clusters_to_instant(
    instant_df: pd.DataFrame,
    scale_results: dict,
    track_col: str = 'unique_id',
    frame_col: str = 'frame'
) -> pd.DataFrame:
    """
    Map clusters from multiple scales to instant dataframe using relative approach.
    Processes 60f scale first (direct window_uid mapping), then other scales relative to 60f.
    """
    print("   🔧 Using relative mapping approach (60f first, others relative)...")
    
    result_df = instant_df.copy()
    
    # Process 60f scale first (if it exists)
    if 'scale_60f' in scale_results:
        print(f"   🔄 Processing scale_60f first (direct window_uid mapping)...")
        scale_info = scale_results['scale_60f']
        result_df = map_clusters_to_instant_relative_to_60f(
            result_df,
            scale_info['windowed_df'],
            scale_info['cluster_labels'],
            'scale_60f'
        )
    
    # Then process other scales relative to 60f
    for scale_name, scale_info in scale_results.items():
        if scale_name == 'scale_60f':
            continue  # Already processed
            
        print(f"   🔄 Processing {scale_name}...")
        result_df = map_clusters_to_instant_relative_to_60f(
            result_df,
            scale_info['windowed_df'],
            scale_info['cluster_labels'],
            scale_name
        )
    
    return result_df
