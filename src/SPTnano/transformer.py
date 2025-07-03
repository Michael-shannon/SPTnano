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
            for start_idx in range(0, len(track_data) - self.window_size + 1, self.step_size):
                end_idx = start_idx + self.window_size
                window_data = track_data.iloc[start_idx:end_idx].copy()
                
                # Extract motion features for this window
                features = self._extract_features(window_data)
                
                if features is not None:
                    # Calculate time_window_num EXACTLY like features.py: start // (window_size - overlap)
                    time_window_num = start_idx // self.step_size
                    
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
    def __init__(self, model, train_dataloader, val_dataloader=None, lr=1e-4, device='cuda', use_tensorboard=False, tensorboard_log_dir=None, augmentation_strategy='basic', use_scheduler=False):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader  # NEW: Add validation dataloader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        
        # Track both training and validation losses
        self.train_losses = []
        self.val_losses = []  # NEW: Track validation losses
        
        # For backward compatibility
        self.losses = self.train_losses
        
        # Learning rate scheduler
        self.use_scheduler = use_scheduler
        self.scheduler = None
        if use_scheduler:
            # Use ReduceLROnPlateau scheduler - reduces LR when loss plateaus
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        
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
    
    def _train_epoch(self, dataloader):
        """Train one epoch and return average loss"""
        self.model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            # Handle both old format (tensor) and new format (dict)
            if isinstance(batch, dict):
                batch_features = batch['features'].to(self.device)
            else:
                batch_features = batch.to(self.device)
            
            x1 = batch_features
            x2 = self.augmentation_fn(batch_features)
            z1 = self.model(x1)
            z2 = self.model(x2)
            loss = contrastive_loss(z1, z2)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def _validate_epoch(self, dataloader):
        """Validate one epoch and return average loss"""
        if dataloader is None:
            return None
            
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle both old format (tensor) and new format (dict)
                if isinstance(batch, dict):
                    batch_features = batch['features'].to(self.device)
                else:
                    batch_features = batch.to(self.device)
                
                x1 = batch_features
                x2 = self.augmentation_fn(batch_features)
                z1 = self.model(x1)
                z2 = self.model(x2)
                loss = contrastive_loss(z1, z2)
                epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def train(self, epochs=10):
        """Enhanced training with validation loss tracking"""
        for epoch in range(epochs):
            # Train epoch
            train_loss = self._train_epoch(self.train_dataloader)
            self.train_losses.append(train_loss)
            
            # Validate epoch
            val_loss = self._validate_epoch(self.val_dataloader)
            if val_loss is not None:
                self.val_losses.append(val_loss)
            
            # Update learning rate scheduler
            if self.use_scheduler and self.scheduler:
                # Use validation loss if available, otherwise training loss
                loss_for_scheduler = val_loss if val_loss is not None else train_loss
                self.scheduler.step(loss_for_scheduler)
            
            # Log to TensorBoard
            if self.use_tensorboard and self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                if val_loss is not None:
                    self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.flush()
            
            # Print progress
            current_lr = self.optimizer.param_groups[0]['lr']
            if val_loss is not None:
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.2e}")
            else:
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, LR = {current_lr:.2e}")
        
        # Close TensorBoard writer
        if self.use_tensorboard and self.writer:
            self.writer.close()

    def plot_loss_curve(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(12, 5))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate if scheduler is used
        if self.use_scheduler and len(self.train_losses) > 1:
            plt.subplot(1, 2, 2)
            # This is approximate - in practice you'd store LR history
            plt.plot(range(len(self.train_losses)), [self.optimizer.param_groups[0]['lr']] * len(self.train_losses))
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def extract_embeddings(self, dataloader=None):
        """Extract embeddings from the dataloader"""
        if dataloader is None:
            dataloader = self.train_dataloader
            
        self.model.eval()
        all_embeddings = []
        track_info = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle both old format (tensor) and new format (dict)
                if isinstance(batch, dict):
                    batch_features = batch['features'].to(self.device)
                    # Extract metadata for each item in the batch
                    batch_metadata = []
                    for i in range(len(batch_features)):
                        metadata = {}
                        for key, value in batch.items():
                            if key != 'features':  # Skip the features tensor
                                if isinstance(value, list):
                                    metadata[key] = value[i] if i < len(value) else None
                                elif hasattr(value, '__getitem__'):
                                    try:
                                        metadata[key] = value[i].item() if hasattr(value[i], 'item') else value[i]
                                    except:
                                        metadata[key] = None
                                else:
                                    metadata[key] = value
                        batch_metadata.append(metadata)
                    track_info.extend(batch_metadata)
                else:
                    batch_features = batch.to(self.device)
                    # No metadata available for old format
                    batch_metadata = [{}] * len(batch_features)
                    track_info.extend(batch_metadata)
                
                embeddings = self.model(batch_features)
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)
    
    def extract_embeddings_with_metadata(self, dataloader=None):
        """Extract embeddings AND metadata for proper mapping"""
        if dataloader is None:
            dataloader = self.train_dataloader
            
        self.model.eval()
        all_embeddings = []
        all_metadata = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Handle both old format (tensor) and new format (dict)
                if isinstance(batch, dict):
                    batch_features = batch['features'].to(self.device)
                    # Extract metadata for each item in the batch
                    for i in range(len(batch_features)):
                        metadata = {'batch_idx': batch_idx, 'within_batch_idx': i}
                        for key, value in batch.items():
                            if key != 'features':  # Skip the features tensor
                                if isinstance(value, list):
                                    metadata[key] = value[i] if i < len(value) else None
                                elif hasattr(value, '__getitem__'):
                                    try:
                                        metadata[key] = value[i].item() if hasattr(value[i], 'item') else value[i]
                                    except:
                                        metadata[key] = None
                                else:
                                    metadata[key] = value
                        all_metadata.append(metadata)
                else:
                    batch_features = batch.to(self.device)
                    # No metadata available for old format
                    for i in range(len(batch_features)):
                        all_metadata.append({'batch_idx': batch_idx, 'within_batch_idx': i})
                
                embeddings = self.model(batch_features)
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings), all_metadata

    def cluster_embeddings(self, embeddings, n_clusters=5):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)

    def save_model(self, path):
        """Save model state dict and training info"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,  # NEW: Save validation losses
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
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        elif 'losses' in checkpoint:  # Backward compatibility
            self.train_losses = checkpoint['losses']
        
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        # Update losses reference for backward compatibility
        self.losses = self.train_losses
        
        # Load augmentation strategy
        if 'augmentation_strategy' in checkpoint:
            self.augmentation_strategy = checkpoint['augmentation_strategy']
            self._setup_augmentation()
        
        print(f"✅ Model loaded from: {path}")
        print(f"   Previous training epochs: {len(self.train_losses)}")
        if self.val_losses:
            print(f"   Previous validation epochs: {len(self.val_losses)}")
        if 'augmentation_strategy' in checkpoint:
            print(f"   Augmentation strategy: {checkpoint['augmentation_strategy']}")
    
    def resume_training(self, epochs=10):
        """Resume training from current state"""
        print(f"🔄 Resuming training from epoch {len(self.train_losses) + 1}")
        self.train(epochs)

# Convenience functions for easy use from notebooks
def create_trajectory_dataset(instant_df, window_size=60, overlap=30, min_track_length=60):
    """Create a trajectory dataset from pandas DataFrame"""
    return PandasTrajectoryDataset(instant_df, window_size, overlap, min_track_length)

def create_smart_train_val_test_split(instant_df, 
                                    val_split=0.15, 
                                    test_split=0.2, 
                                    split_strategy='hybrid_cell',
                                    random_seed=42):
    """
    Create smart train/validation/test splits with multiple strategies.
    
    Args:
        instant_df: DataFrame with trajectory data
        val_split: Fraction for validation set
        test_split: Fraction for test set  
        split_strategy: 'hybrid_cell', 'stratified', 'random', 'cell_balanced'
        random_seed: Random seed for reproducibility
    
    Returns:
        train_df, val_df, test_df: Split dataframes
        split_info: Dictionary with split statistics
    """
    np.random.seed(random_seed)
    
    # Get track and cell information
    track_info = instant_df.groupby('unique_id').agg({
        'condition': 'first',
        'filename': 'first',  # Cell identifier
        'frame': 'count'      # Track length
    }).reset_index()
    track_info.columns = ['unique_id', 'condition', 'filename', 'track_length']
    
    print(f"📊 Data Overview:")
    print(f"   Total tracks: {len(track_info)}")
    print(f"   Unique conditions: {track_info['condition'].nunique()}")
    print(f"   Unique cells (filenames): {track_info['filename'].nunique()}")
    print(f"   Strategy: {split_strategy}")
    
    if split_strategy == 'hybrid_cell':
        # 🎯 HYBRID STRATEGY: Mixed train/val, complete cells for test
        print(f"\n🎯 Hybrid Cell Strategy:")
        print(f"   - Test set: Complete cells (clean visualization)")
        print(f"   - Train/Val: Mixed tracks from remaining cells (robust training)")
        
        # Group tracks by cell (filename) and condition
        cells_info = track_info.groupby('filename').agg({
            'unique_id': 'count',
            'condition': lambda x: list(x.unique()),
            'track_length': 'sum'
        }).reset_index()
        cells_info.columns = ['filename', 'n_tracks', 'conditions', 'total_frames']
        
        # 🔥 NEW: Condition-aware test cell selection
        print(f"\n🔍 Analyzing cells by condition for balanced test selection...")
        
        # Get unique conditions and their distribution
        all_conditions = track_info['condition'].unique()
        condition_track_counts = track_info['condition'].value_counts()
        
        print(f"   Available conditions: {list(all_conditions)}")
        print(f"   Track distribution: {dict(condition_track_counts)}")
        
        # Group cells by their primary condition (handle multi-condition cells)
        cells_by_condition = {}
        for _, cell in cells_info.iterrows():
            # Handle cells that might have multiple conditions (should be rare)
            primary_condition = cell['conditions'][0] if cell['conditions'] else 'unknown'
            if len(cell['conditions']) > 1:
                print(f"   ⚠️ Cell {cell['filename']} has multiple conditions: {cell['conditions']}")
                print(f"      Using primary condition: {primary_condition}")
            
            if primary_condition not in cells_by_condition:
                cells_by_condition[primary_condition] = []
            cells_by_condition[primary_condition].append(cell)
        
        # Calculate target test tracks per condition (proportional to condition frequency)
        target_test_tracks = int(len(track_info) * test_split)
        target_per_condition = {}
        
        for condition in all_conditions:
            condition_proportion = condition_track_counts[condition] / len(track_info)
            target_per_condition[condition] = int(target_test_tracks * condition_proportion)
        
        print(f"   Target test tracks per condition: {target_per_condition}")
        
        # Select test cells while balancing conditions
        test_cells = []
        selected_per_condition = {condition: 0 for condition in all_conditions}
        
        # Sort cells within each condition by total frames (prefer larger cells for cleaner visualization)
        for condition in all_conditions:
            if condition in cells_by_condition:
                condition_cells = sorted(cells_by_condition[condition], 
                                       key=lambda x: x['total_frames'], reverse=True)
                
                # Select cells for this condition
                for cell in condition_cells:
                    current_selected = selected_per_condition[condition]
                    target_for_condition = target_per_condition[condition]
                    
                    # Add cell if we haven't reached the target for this condition
                    # Or if no cells selected yet for this condition (ensure each condition gets at least one cell)
                    if (current_selected < target_for_condition or 
                        (current_selected == 0 and len(test_cells) < len(all_conditions))):
                        
                        test_cells.append(cell['filename'])
                        selected_per_condition[condition] += cell['n_tracks']
                        
                        print(f"   ✅ Selected cell {cell['filename']} ({condition}): {cell['n_tracks']} tracks")
                        
                        # Check if we have enough total test tracks
                        total_selected = sum(selected_per_condition.values())
                        if total_selected >= target_test_tracks:
                            break
                
                # Break outer loop if we have enough tracks
                total_selected = sum(selected_per_condition.values())
                if total_selected >= target_test_tracks:
                    break
        
        # Fallback: if we don't have enough cells, add more from any condition
        total_selected = sum(selected_per_condition.values())
        if total_selected < target_test_tracks * 0.8:  # If we're significantly under target
            print(f"   ⚠️ Only selected {total_selected} tracks, target was {target_test_tracks}")
            print(f"   📋 Adding additional cells to reach target...")
            
            # Get all remaining cells
            all_remaining_cells = []
            for condition, cells in cells_by_condition.items():
                for cell in cells:
                    if cell['filename'] not in test_cells:
                        all_remaining_cells.append(cell)
            
            # Sort by size and add until we reach target
            all_remaining_cells.sort(key=lambda x: x['total_frames'], reverse=True)
            for cell in all_remaining_cells:
                if total_selected >= target_test_tracks:
                    break
                test_cells.append(cell['filename'])
                # Find which condition this cell belongs to
                cell_condition = cell['conditions'][0]
                selected_per_condition[cell_condition] += cell['n_tracks']
                total_selected += cell['n_tracks']
                print(f"   📋 Added {cell['filename']} ({cell_condition}): {cell['n_tracks']} tracks")
        
        # Report final test set composition
        print(f"\n📊 Final Test Set Composition:")
        for condition in all_conditions:
            selected = selected_per_condition[condition]
            total_condition = condition_track_counts[condition]
            percentage = (selected / total_condition * 100) if total_condition > 0 else 0
            print(f"   {condition}: {selected}/{total_condition} tracks ({percentage:.1f}%)")
        
        # Get test tracks (complete cells)
        test_track_ids = track_info[track_info['filename'].isin(test_cells)]['unique_id'].tolist()
        
        # Remaining tracks for train/val (mixed cells)
        remaining_track_info = track_info[~track_info['filename'].isin(test_cells)]
        
        # Stratified split of remaining tracks for train/val
        train_tracks, val_tracks = _stratified_track_split(
            remaining_track_info, 
            val_fraction=val_split / (1 - test_split),  # Adjust for reduced pool
            random_seed=random_seed
        )
        
        split_info = {
            'strategy': 'hybrid_cell',
            'test_cells': test_cells,
            'test_tracks': len(test_track_ids),
            'train_tracks': len(train_tracks),
            'val_tracks': len(val_tracks),
            'test_complete_cells': True,
            'test_condition_balance': selected_per_condition,  # NEW: Track condition balance
            'target_per_condition': target_per_condition  # NEW: Track targets
        }
        
    elif split_strategy == 'stratified':
        # Stratified by condition only
        train_tracks, remaining_tracks = _stratified_track_split(
            track_info, 
            val_fraction=val_split + test_split,
            random_seed=random_seed
        )
        
        remaining_info = track_info[track_info['unique_id'].isin(remaining_tracks)]
        val_tracks, test_track_ids = _stratified_track_split(
            remaining_info,
            val_fraction=val_split / (val_split + test_split),
            random_seed=random_seed + 1
        )
        
        split_info = {
            'strategy': 'stratified',
            'test_complete_cells': False
        }
        
    elif split_strategy == 'cell_balanced':
        # Balance cells across splits
        cells_info = track_info.groupby('filename').agg({
            'unique_id': list,
            'condition': lambda x: list(x.unique())
        }).reset_index()
        
        # Distribute cells across splits
        np.random.shuffle(cells_info.values)
        n_cells = len(cells_info)
        
        test_cells_end = int(n_cells * test_split)
        val_cells_end = test_cells_end + int(n_cells * val_split)
        
        test_cells = cells_info.iloc[:test_cells_end]
        val_cells = cells_info.iloc[test_cells_end:val_cells_end]
        train_cells = cells_info.iloc[val_cells_end:]
        
        test_track_ids = [track for tracks in test_cells['unique_id'] for track in tracks]
        val_tracks = [track for tracks in val_cells['unique_id'] for track in tracks]
        train_tracks = [track for tracks in train_cells['unique_id'] for track in tracks]
        
        split_info = {
            'strategy': 'cell_balanced',
            'test_complete_cells': True
        }
        
    else:  # random
        # Simple random split
        all_tracks = track_info['unique_id'].tolist()
        np.random.shuffle(all_tracks)
        
        n_test = int(len(all_tracks) * test_split)
        n_val = int(len(all_tracks) * val_split)
        
        test_track_ids = all_tracks[:n_test]
        val_tracks = all_tracks[n_test:n_test + n_val]
        train_tracks = all_tracks[n_test + n_val:]
        
        split_info = {
            'strategy': 'random',
            'test_complete_cells': False
        }
    
    # Create the split dataframes
    train_df = instant_df[instant_df['unique_id'].isin(train_tracks)].copy()
    val_df = instant_df[instant_df['unique_id'].isin(val_tracks)].copy()
    test_df = instant_df[instant_df['unique_id'].isin(test_track_ids)].copy()
    
    # Update split info with final statistics
    split_info.update({
        'total_tracks': len(track_info),
        'train_tracks': len(train_tracks),
        'val_tracks': len(val_tracks),
        'test_tracks': len(test_track_ids),
        'train_points': len(train_df),
        'val_points': len(val_df),
        'test_points': len(test_df)
    })
    
    # Report split quality
    print(f"\n✅ Split Results:")
    print(f"   Train: {len(train_tracks)} tracks ({len(train_df):,} points)")
    print(f"   Val:   {len(val_tracks)} tracks ({len(val_df):,} points)")
    print(f"   Test:  {len(test_track_ids)} tracks ({len(test_df):,} points)")
    
    # Check condition balance
    print(f"\n🔍 Condition Balance:")
    for split_name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        if len(df) > 0:
            condition_counts = df.groupby('unique_id')['condition'].first().value_counts()
            print(f"   {split_name}: {dict(condition_counts)}")
    
    if split_strategy == 'hybrid_cell':
        print(f"\n🎯 Test Set Cell Info:")
        test_cell_info = test_df.groupby('filename').agg({
            'unique_id': 'nunique',
            'condition': lambda x: list(x.unique())[0]
        })
        for filename, info in test_cell_info.iterrows():
            print(f"   {filename}: {info['unique_id']} tracks, {info['condition']}")
        
        # NEW: Show condition balance quality
        if 'test_condition_balance' in split_info:
            print(f"\n🎯 Test Set Condition Balance Quality:")
            actual_balance = split_info['test_condition_balance']
            target_balance = split_info['target_per_condition']
            
            for condition in actual_balance.keys():
                actual = actual_balance[condition]
                target = target_balance[condition]
                ratio = (actual / target) if target > 0 else float('inf')
                
                if 0.8 <= ratio <= 1.2:  # Within 20% of target
                    status = "✅ Good"
                elif 0.5 <= ratio <= 2.0:  # Within 50% of target
                    status = "⚠️ Acceptable"
                else:
                    status = "❌ Imbalanced"
                
                print(f"   {condition}: {actual}/{target} tracks (ratio: {ratio:.2f}) {status}")
    
    return train_df, val_df, test_df, split_info

def _stratified_track_split(track_info, val_fraction, random_seed=42):
    """Helper function for stratified splitting by condition"""
    np.random.seed(random_seed)
    
    train_tracks = []
    val_tracks = []
    
    for condition in track_info['condition'].unique():
        condition_tracks = track_info[track_info['condition'] == condition]['unique_id'].tolist()
        np.random.shuffle(condition_tracks)
        
        n_val = int(len(condition_tracks) * val_fraction)
        val_tracks.extend(condition_tracks[:n_val])
        train_tracks.extend(condition_tracks[n_val:])
    
    return train_tracks, val_tracks

def train_motion_transformer(instant_df, 
                           window_size=60, 
                           overlap=30, 
                           batch_size=64, 
                           epochs=10, 
                           val_split=0.15,
                           test_split=0.2,
                           split_strategy='hybrid_cell',
                           device='auto',
                           use_tensorboard=False,
                           tensorboard_log_dir=None,
                           augmentation_strategy='basic',
                           save_model_path=None,
                           use_scheduler=False):
    """
    Enhanced pipeline to train a motion transformer with smart data splitting and validation tracking.
    
    Args:
        instant_df: DataFrame with trajectory data
        window_size: Number of frames per window
        overlap: Overlap between windows
        batch_size: Training batch size
        epochs: Number of training epochs
        val_split: Fraction of data for validation (from train+val pool)
        test_split: Fraction of data for testing
        split_strategy: 'hybrid_cell', 'stratified', 'random', 'cell_balanced'
        device: 'auto', 'cuda', or 'cpu'
        use_tensorboard: Enable TensorBoard logging
        tensorboard_log_dir: Directory for TensorBoard logs
        augmentation_strategy: Augmentation strategy
        save_model_path: Path to save trained model
        use_scheduler: Use learning rate scheduler
    
    Returns:
        trainer: Trained MotionTrainer object
        datasets: Dict with train, val, test datasets
        split_info: Information about the data split
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🚀 Enhanced Motion Transformer Training")
    print(f"   Device: {device}")
    print(f"   Window: {window_size}f, Overlap: {overlap}f")
    print(f"   Splits: {1-(val_split+test_split):.1f} train, {val_split:.1f} val, {test_split:.1f} test")
    
    # Create smart train/val/test split
    train_df, val_df, test_df, split_info = create_smart_train_val_test_split(
        instant_df, 
        val_split=val_split,
        test_split=test_split, 
        split_strategy=split_strategy
    )
    
    # Create datasets
    print(f"\n🔄 Creating datasets...")
    train_dataset = PandasTrajectoryDataset(train_df, window_size, overlap)
    val_dataset = PandasTrajectoryDataset(val_df, window_size, overlap) if len(val_df) > 0 else None
    test_dataset = PandasTrajectoryDataset(test_df, window_size, overlap)
    
    print(f"   Train windows: {len(train_dataset)}")
    print(f"   Val windows: {len(val_dataset) if val_dataset else 0}")
    print(f"   Test windows: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create and train model
    model = TransformerMotionEncoder(input_dim=3)
    trainer = MotionTrainer(
        model, 
        train_loader, 
        val_loader,  # Now properly supports validation
        device=device, 
        use_tensorboard=use_tensorboard, 
        tensorboard_log_dir=tensorboard_log_dir,
        augmentation_strategy=augmentation_strategy,
        use_scheduler=use_scheduler
    )
    
    print(f"\n🚀 Starting training with validation tracking...")
    trainer.train(epochs=epochs)
    
    # Save model if path provided
    if save_model_path:
        trainer.save_model(save_model_path)
    
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    
    return trainer, datasets, split_info

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
            
            trainer, datasets, split_info = train_motion_transformer(
                df,
                window_size=window_size,
                overlap=overlap,
                batch_size=batch_size,
                epochs=epochs,
                val_split=0.15,  # Add validation split
                test_split=test_split,
                split_strategy='hybrid_cell',  # Use smart splitting
                device=device,
                use_tensorboard=use_tensorboard,
                tensorboard_log_dir=scale_tensorboard_dir,
                augmentation_strategy=augmentation_strategy,
                save_model_path=scale_model_path,
                use_scheduler=use_scheduler
            )
            
            train_dataset = datasets['train']
            test_dataset = datasets['test']
            
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

def map_clusters_to_time_windowed_df(
    time_windowed_df: pd.DataFrame, 
    transformer_windows: list, 
    cluster_labels: np.ndarray
) -> pd.DataFrame:
    """
    Step 1: Map transformer clusters to time_windowed_df using window_uid.
    
    This is the direct mapping step - transformer windows should have the same
    window_uid format as time_windowed_df: unique_id_timewindow_framestart_frameend
    
    Parameters:
    -----------
    time_windowed_df : pd.DataFrame
        Your original time_windowed_df with window_uid column
    transformer_windows : list
        List of window info dicts from transformer dataset.get_track_info()
    cluster_labels : np.ndarray
        Cluster assignments from clustering the transformer embeddings
        
    Returns:
    --------
    pd.DataFrame
        time_windowed_df with cluster column added
    """
    print("🔗 Step 1: Mapping clusters to time_windowed_df using window_uid...")
    print("🔍 DETAILED DIAGNOSTICS:")
    
    # Create window_uid -> cluster mapping
    window_cluster_map = {}
    for i, window_info in enumerate(transformer_windows):
        if i < len(cluster_labels):
            window_uid = window_info['window_uid']
            cluster_id = cluster_labels[i]
            window_cluster_map[window_uid] = cluster_id
    
    print(f"   📊 Created mapping for {len(window_cluster_map)} transformer windows")
    
    # Diagnostic: Check window_uid formats
    print(f"\n🔍 WINDOW_UID FORMAT ANALYSIS:")
    
    # Sample transformer window_uids
    transformer_sample = list(window_cluster_map.keys())[:5]
    print(f"   📋 Sample transformer window_uids:")
    for uid in transformer_sample:
        print(f"      • {uid}")
    
    # Sample time_windowed_df window_uids
    windowed_sample = time_windowed_df['window_uid'].head(5).tolist()
    print(f"   📋 Sample time_windowed_df window_uids:")
    for uid in windowed_sample:
        print(f"      • {uid}")
    
    # Check for exact matches
    transformer_uids = set(window_cluster_map.keys())
    windowed_uids = set(time_windowed_df['window_uid'].tolist())
    
    matches = transformer_uids.intersection(windowed_uids)
    transformer_only = transformer_uids - windowed_uids
    windowed_only = windowed_uids - transformer_uids
    
    print(f"\n🎯 MATCHING ANALYSIS:")
    print(f"   ✅ Exact matches: {len(matches)}")
    print(f"   🔴 Transformer only: {len(transformer_only)}")
    print(f"   🔵 Time_windowed only: {len(windowed_only)}")
    
    if len(transformer_only) > 0:
        print(f"   📋 Sample transformer-only window_uids:")
        for uid in list(transformer_only)[:3]:
            print(f"      • {uid}")
    
    if len(windowed_only) > 0:
        print(f"   📋 Sample windowed-only window_uids:")
        for uid in list(windowed_only)[:3]:
            print(f"      • {uid}")
    
    # Add cluster column to time_windowed_df
    result_df = time_windowed_df.copy()
    result_df['cluster'] = result_df['window_uid'].map(window_cluster_map)
    
    # Report mapping success
    mapped_count = result_df['cluster'].notna().sum()
    total_count = len(result_df)
    coverage = (mapped_count / total_count) * 100
    
    print(f"\n   ✅ Successfully mapped {mapped_count:,}/{total_count:,} windows ({coverage:.1f}% coverage)")
    
    return result_df

def map_clusters_from_windowed_to_instant_df(
    instant_df: pd.DataFrame, 
    time_windowed_df_with_clusters: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 2: Map clusters from time_windowed_df to instant_df using existing window_uid.
    
    This uses the pre-calculated window_uid mapping that already exists between
    time_windowed_df and instant_df (created when features were calculated).
    
    Parameters:
    -----------
    instant_df : pd.DataFrame
        Your original instant_df with window_uid column
    time_windowed_df_with_clusters : pd.DataFrame
        time_windowed_df with cluster column added from step 1
        
    Returns:
    --------
    pd.DataFrame
        instant_df with cluster column added
    """
    print("🔗 Step 2: Mapping clusters from time_windowed_df to instant_df using window_uid...")
    
    # Create window_uid -> cluster mapping from time_windowed_df
    window_cluster_map = {}
    for _, row in time_windowed_df_with_clusters.iterrows():
        window_uid = row['window_uid']
        cluster_id = row['cluster']
        if pd.notna(cluster_id):
            window_cluster_map[window_uid] = cluster_id
    
    print(f"   📊 Created mapping for {len(window_cluster_map)} windows with clusters")
    
    # Add cluster column to instant_df using existing window_uid
    result_df = instant_df.copy()
    result_df['cluster'] = result_df['window_uid'].map(window_cluster_map)
    
    # Report mapping success
    mapped_count = result_df['cluster'].notna().sum()
    total_count = len(result_df)
    coverage = (mapped_count / total_count) * 100
    
    print(f"   ✅ Successfully mapped {mapped_count:,}/{total_count:,} trajectory points ({coverage:.1f}% coverage)")
    
    return result_df

def cluster_test_set_only_with_mapping(
    trainer,
    datasets: dict,
    time_windowed_df: pd.DataFrame,
    instant_df: pd.DataFrame,
    n_clusters: int = 5,
    cluster_method: str = 'kmeans'
) -> dict:
    """
    Complete test-set-only pipeline: Extract test embeddings, cluster, and map to filtered dataframes.
    
    This is the scientifically rigorous approach that:
    1. Clusters TEST embeddings only
    2. Filters time_windowed_df to TEST cells only
    3. Filters instant_df to TEST cells only  
    4. Maps test clusters to test-only dataframes using window_uid
    
    Parameters:
    -----------
    trainer : MotionTrainer
        Trained transformer model
    datasets : dict
        Dictionary containing 'train', 'val', 'test' datasets
    time_windowed_df : pd.DataFrame
        Full time_windowed_df (will be filtered to test cells)
    instant_df : pd.DataFrame
        Full instant_df (will be filtered to test cells)
    n_clusters : int
        Number of clusters
    cluster_method : str
        Clustering method ('kmeans', 'hdbscan')
        
    Returns:
    --------
    dict
        Results with test-only dataframes containing cluster assignments
    """
    print("🧪 TEST-SET-ONLY Transformer Clustering with window_uid Mapping")
    print("=" * 70)
    
    # Step 1: Extract test embeddings and cluster
    print("📊 Step 1: Extracting and clustering TEST embeddings...")
    from torch.utils.data import DataLoader
    
    test_dataset = datasets['test']
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    test_embeddings = trainer.extract_embeddings(test_dataloader)
    test_clusters = trainer.cluster_embeddings(test_embeddings, n_clusters=n_clusters)
    transformer_windows = test_dataset.get_track_info()
    
    print(f"   ✅ Extracted {test_embeddings.shape[0]} test embeddings")
    print(f"   ✅ Clustered into {len(np.unique(test_clusters))} clusters")
    
    # Step 2: Get test cell filenames
    print("\n📂 Step 2: Identifying test cells...")
    test_cell_filenames = set()
    for window_info in transformer_windows:
        # Get track info from the original dataset to find filename
        track_id = window_info['unique_id']
        track_condition = window_info['condition']
        test_cell_filenames.add((track_id, track_condition))
    
    # Get actual filenames from instant_df for these tracks
    test_track_ids = [info['unique_id'] for info in transformer_windows]
    test_filenames = instant_df[instant_df['unique_id'].isin(test_track_ids)]['filename'].unique()
    
    print(f"   ✅ Test tracks: {len(test_track_ids)} unique tracks")
    print(f"   ✅ Test cells: {len(test_filenames)} unique filenames")
    print(f"   📋 Test filenames: {list(test_filenames)[:5]}{'...' if len(test_filenames) > 5 else ''}")
    
    # Step 3: Filter time_windowed_df to test cells only
    print("\n🔍 Step 3: Filtering time_windowed_df to test cells only...")
    if 'filename' in time_windowed_df.columns:
        test_time_windowed_df = time_windowed_df[time_windowed_df['filename'].isin(test_filenames)].copy()
    else:
        # Fallback: filter by unique_id
        test_time_windowed_df = time_windowed_df[time_windowed_df['unique_id'].isin(test_track_ids)].copy()
    
    print(f"   📊 Original time_windowed_df: {len(time_windowed_df)} windows")
    print(f"   📊 Test-only time_windowed_df: {len(test_time_windowed_df)} windows")
    print(f"   📊 Reduction: {len(time_windowed_df) - len(test_time_windowed_df)} windows filtered out")
    
    # Step 4: Filter instant_df to test cells only
    print("\n🔍 Step 4: Filtering instant_df to test cells only...")
    test_instant_df = instant_df[instant_df['filename'].isin(test_filenames)].copy()
    
    print(f"   📊 Original instant_df: {len(instant_df)} trajectory points")
    print(f"   📊 Test-only instant_df: {len(test_instant_df)} trajectory points")
    print(f"   📊 Reduction: {len(instant_df) - len(test_instant_df)} points filtered out")
    
    # Step 5: Map test clusters to test-only time_windowed_df
    print("\n🔗 Step 5: Mapping clusters to test-only time_windowed_df...")
    
    # Create window_uid -> cluster mapping from transformer results
    window_cluster_map = {}
    for i, window_info in enumerate(transformer_windows):
        window_uid = window_info['window_uid']
        cluster_id = test_clusters[i]
        window_cluster_map[window_uid] = cluster_id
    
    print(f"   📊 Created mapping for {len(window_cluster_map)} test windows")
    
    # Add cluster column to test time_windowed_df
    test_time_windowed_df_with_clusters = test_time_windowed_df.copy()
    test_time_windowed_df_with_clusters['cluster'] = test_time_windowed_df_with_clusters['window_uid'].map(window_cluster_map)
    
    # Report mapping success for windowed data
    windowed_mapped_count = test_time_windowed_df_with_clusters['cluster'].notna().sum()
    windowed_total_count = len(test_time_windowed_df_with_clusters)
    windowed_coverage = (windowed_mapped_count / windowed_total_count) * 100
    
    print(f"   ✅ Windowed mapping: {windowed_mapped_count:,}/{windowed_total_count:,} windows ({windowed_coverage:.1f}% coverage)")
    
    # Step 6: Map clusters from test windowed to test instant using window_uid
    print("\n🔗 Step 6: Mapping clusters from test windowed to test instant...")
    
    # Create window_uid -> cluster mapping from test windowed data
    windowed_cluster_map = {}
    for _, row in test_time_windowed_df_with_clusters.iterrows():
        window_uid = row['window_uid']
        cluster_id = row['cluster']
        if pd.notna(cluster_id):
            windowed_cluster_map[window_uid] = cluster_id
    
    # Add cluster column to test instant_df
    test_instant_df_with_clusters = test_instant_df.copy()
    test_instant_df_with_clusters['cluster'] = test_instant_df_with_clusters['window_uid'].map(windowed_cluster_map)
    
    # Report mapping success for instant data
    instant_mapped_count = test_instant_df_with_clusters['cluster'].notna().sum()
    instant_total_count = len(test_instant_df_with_clusters)
    instant_coverage = (instant_mapped_count / instant_total_count) * 100
    
    print(f"   ✅ Instant mapping: {instant_mapped_count:,}/{instant_total_count:,} points ({instant_coverage:.1f}% coverage)")
    
    # Step 7: Calculate clustering quality metrics
    print("\n📊 Step 7: Calculating clustering quality metrics...")
    
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    import collections
    
    test_silhouette = silhouette_score(test_embeddings, test_clusters)
    test_davies_bouldin = davies_bouldin_score(test_embeddings, test_clusters)
    cluster_counts = collections.Counter(test_clusters)
    cluster_balance = min(cluster_counts.values()) / max(cluster_counts.values())
    
    print(f"   • Silhouette Score: {test_silhouette:.3f}")
    print(f"   • Davies-Bouldin Score: {test_davies_bouldin:.3f}")
    print(f"   • Cluster Balance: {cluster_balance:.3f}")
    print(f"   • Cluster Counts: {dict(cluster_counts)}")
    
    # Final success assessment
    mapping_success = windowed_coverage > 80 and instant_coverage > 50
    
    print(f"\n🎯 TEST-SET-ONLY CLUSTERING RESULTS:")
    print(f"   ✅ Test embeddings: {len(test_embeddings)} windows from {len(test_filenames)} cells")
    print(f"   ✅ Windowed coverage: {windowed_coverage:.1f}%")
    print(f"   ✅ Instant coverage: {instant_coverage:.1f}%")
    print(f"   ✅ Mapping success: {'✅ SUCCESS' if mapping_success else '❌ FAILED'}")
    
    if mapping_success:
        print(f"   🚀 Ready for visualization and analysis!")
    else:
        print(f"   ⚠️  Low coverage - check window_uid consistency")
    
    return {
        'test_time_windowed_df_with_clusters': test_time_windowed_df_with_clusters,
        'test_instant_df_with_clusters': test_instant_df_with_clusters,
        'test_embeddings': test_embeddings,
        'test_clusters': test_clusters,
        'test_filenames': test_filenames,
        'transformer_windows': transformer_windows,
        'cluster_info': {
            'silhouette_score': test_silhouette,
            'davies_bouldin_score': test_davies_bouldin,
            'cluster_balance': cluster_balance,
            'cluster_counts': cluster_counts,
            'windowed_coverage': windowed_coverage,
            'instant_coverage': instant_coverage,
            'mapping_success': mapping_success,
            'n_test_cells': len(test_filenames),
            'n_test_windows': len(test_embeddings),
            'n_test_points': len(test_instant_df_with_clusters)
        }
    }
