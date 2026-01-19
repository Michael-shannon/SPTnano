import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class TransformerMotionEncoder(nn.Module):
    def __init__(
        self, input_dim=3, embed_dim=64, num_heads=4, ff_dim=128, num_layers=2
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        
        # Store attention weights if needed (for interpretability)
        self.store_attention = False
        self.attention_weights = None

    def forward(self, x, return_attention=False):  # x: [B, T, 3]
        """
        Forward pass with optional attention weight extraction.
        
        Args:
            x: Input tensor [B, T, input_dim]
            return_attention: If True, also return attention weights (slower)
            
        Returns:
            embeddings: [B, embed_dim]
            attention_weights (optional): List of attention weight tensors per layer
        """
        B, T, _ = x.shape
        x = self.input_proj(x)  # [B, T, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, T+1, embed_dim]
        x = self.transformer_encoder(x.permute(1, 0, 2))  # [T+1, B, embed_dim]
        x = x[0]  # take CLS token [B, embed_dim]
        
        # Note: Standard PyTorch TransformerEncoder doesn't expose attention weights
        # For full attention extraction, you'd need to use a custom encoder
        # or access layer internals during forward pass
        
        if return_attention:
            # This is a placeholder - actual implementation would need custom encoder
            # See extract_attention_weights() method for practical alternatives
            print("⚠️ Standard TransformerEncoder doesn't expose attention weights")
            print("💡 Use extract_attention_weights() method for interpretability")
            return self.norm(x), None
        
        return self.norm(x)
    
    def extract_attention_weights(self, x, method='gradient'):
        """
        Extract interpretable attention/importance scores from the model.
        
        This provides alternatives to true attention weights for understanding
        what the model focuses on.
        
        Args:
            x: Input tensor [B, T, input_dim]
            method: 'gradient' or 'leave_one_out'
            
        Returns:
            importance_scores: [B, T] tensor showing frame importance
        """
        if method == 'gradient':
            return self._gradient_based_importance(x)
        elif method == 'leave_one_out':
            return self._leave_one_out_importance(x)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _gradient_based_importance(self, x):
        """
        Compute frame importance using input gradients.
        Shows which frames, if perturbed, would most change the embedding.
        """
        self.eval()
        x_requires_grad = x.clone().requires_grad_(True)
        
        # Forward pass
        embedding = self.forward(x_requires_grad)
        
        # Compute gradient of embedding norm w.r.t. input
        embedding_norm = embedding.norm(dim=1).sum()
        embedding_norm.backward()
        
        # Gradient magnitude = importance
        importance = x_requires_grad.grad.abs().mean(dim=2)  # [B, T]
        
        return importance.detach()
    
    def _leave_one_out_importance(self, x):
        """
        Compute frame importance using leave-one-out analysis.
        Shows which frames, if removed, would most change the embedding.
        """
        self.eval()
        B, T, C = x.shape
        
        with torch.no_grad():
            # Get baseline embedding
            embedding_full = self.forward(x)
            
            # Test removing each frame
            importance_scores = []
            for t in range(T):
                # Mask out frame t
                x_masked = x.clone()
                x_masked[:, t, :] = 0
                
                # Get embedding without this frame
                embedding_masked = self.forward(x_masked)
                
                # Compute change in embedding
                importance = torch.norm(embedding_full - embedding_masked, dim=1)
                importance_scores.append(importance)
            
            importance_scores = torch.stack(importance_scores, dim=1)  # [B, T]
        
        return importance_scores


class TimeWindowDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)  # [N, 60, 3]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PandasTrajectoryDataset(Dataset):
    """
    Dataset class for extracting time windows from trajectory dataframe.
    Creates sequences of (dx, dy, direction) motion features.
    Supports both Pandas and Polars DataFrames natively.
    """

    def __init__(self, instant_df, window_size=60, overlap=30, min_track_length=60):
        """
        Args:
            instant_df: DataFrame with trajectory data (Pandas or Polars)
            window_size: Number of frames per window
            overlap: Number of overlapping frames between windows
            min_track_length: Minimum track length to consider

        """
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = window_size - overlap
        self.min_track_length = min_track_length
        self.is_polars = isinstance(instant_df, pl.DataFrame)

        # Sort by unique_id and frame to ensure temporal order
        if self.is_polars:
            self.df = instant_df.sort(["unique_id", "frame"])
        else:
            self.df = instant_df.sort_values(["unique_id", "frame"]).reset_index(drop=True)

        # Generate windows
        self.windows = self._generate_windows()

    def _generate_windows(self):
        """Generate all valid windows from the trajectory data"""
        windows = []

        if self.is_polars:
            # Use Polars native operations
            for unique_id in self.df["unique_id"].unique():
                track_data = self.df.filter(pl.col("unique_id") == unique_id).sort("frame")
                
                # Skip short tracks
                if len(track_data) < self.min_track_length:
                    continue

                # Convert to pandas for window extraction (small per-track conversion)
                track_pandas = track_data.to_pandas()
                
                # Generate overlapping windows for this track
                for start_idx in range(
                    0, len(track_pandas) - self.window_size + 1, self.step_size
                ):
                    end_idx = start_idx + self.window_size
                    window_data = track_pandas.iloc[start_idx:end_idx].copy()

                    # Extract motion features for this window
                    features = self._extract_features(window_data)

                    if features is not None:
                        # Calculate time_window_num EXACTLY like features.py: start // (window_size - overlap)
                        time_window_num = start_idx // self.step_size

                        # Create window_uid that matches features.py format: unique_id_timewindow_framestart_frameend
                        frame_start = window_data.iloc[0]["frame"]
                        frame_end = window_data.iloc[-1]["frame"]
                        window_uid = (
                            f"{unique_id}_{time_window_num}_{frame_start}_{frame_end}"
                        )

                        windows.append(
                            {
                                "features": features,
                                "unique_id": unique_id,
                                "start_frame": frame_start,
                                "end_frame": frame_end,
                                "window_idx": len(windows),
                                "window_uid": window_uid,
                                "time_window": time_window_num,
                                "condition": window_data["condition"].iloc[0],
                            }
                        )
        else:
            # Use Pandas operations
            for unique_id, track_data in self.df.groupby("unique_id"):
                track_data = track_data.sort_values("frame").reset_index(drop=True)

                # Skip short tracks
                if len(track_data) < self.min_track_length:
                    continue

                # Generate overlapping windows for this track
                for start_idx in range(
                    0, len(track_data) - self.window_size + 1, self.step_size
                ):
                    end_idx = start_idx + self.window_size
                    window_data = track_data.iloc[start_idx:end_idx].copy()

                    # Extract motion features for this window
                    features = self._extract_features(window_data)

                    if features is not None:
                        # Calculate time_window_num EXACTLY like features.py: start // (window_size - overlap)
                        time_window_num = start_idx // self.step_size

                        # Create window_uid that matches features.py format: unique_id_timewindow_framestart_frameend
                        frame_start = window_data.iloc[0]["frame"]
                        frame_end = window_data.iloc[-1]["frame"]
                        window_uid = (
                            f"{unique_id}_{time_window_num}_{frame_start}_{frame_end}"
                        )

                        windows.append(
                            {
                                "features": features,
                                "unique_id": unique_id,
                                "start_frame": frame_start,
                                "end_frame": frame_end,
                                "window_idx": len(windows),
                                "window_uid": window_uid,
                                "time_window": time_window_num,
                                "condition": track_data["condition"].iloc[0],
                            }
                        )

        return windows

    def _extract_features(self, window_data):
        """
        Extract motion features from a window of trajectory data.
        Returns array of shape (window_size, 3) with features: [dx, dy, direction]
        """
        try:
            # Calculate dx, dy (displacement between consecutive points)
            dx = np.diff(
                window_data["x_um"].values, prepend=window_data["x_um"].iloc[0]
            )
            dy = np.diff(
                window_data["y_um"].values, prepend=window_data["y_um"].iloc[0]
            )

            # Calculate direction
            if "direction_rad" in window_data.columns:
                direction = window_data["direction_rad"].values
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
            "features": torch.FloatTensor(window["features"]),
            "unique_id": window["unique_id"],
            "start_frame": window["start_frame"],
            "end_frame": window["end_frame"],
            "window_idx": window["window_idx"],
            "window_uid": window["window_uid"],
            "time_window": window["time_window"],
            "condition": window["condition"],
        }

    def get_labels(self):
        """Return condition labels for each window (for supervised learning if needed)"""
        return [window["condition"] for window in self.windows]

    def get_track_info(self):
        """Return track information for mapping back to original data"""
        return [
            {
                "unique_id": window["unique_id"],
                "start_frame": window["start_frame"],
                "end_frame": window["end_frame"],
                "window_uid": window[
                    "window_uid"
                ],  # NEW: Include window_uid for mapping
                "time_window": window["time_window"],  # NEW: Include time window number
                "condition": window["condition"],
            }
            for window in self.windows
        ]


class TimeAwareTrajectoryDataset(Dataset):
    """
    Memory-efficient dataset for time-aware contrastive learning using POLARS.
    
    Designed for temporal contrastive learning where positive pairs are 
    temporally adjacent windows from the same track (window_t, window_t+1).
    
    Key optimizations:
    1. Keep data in Polars - much more memory efficient than pandas
    2. Store only indices, extract features on-the-fly
    3. Pre-extract coordinate arrays for fast numpy slicing
    4. No data copying during window extraction
    
    Parameters
    ----------
    instant_df : pl.DataFrame or pd.DataFrame
        DataFrame with trajectory data containing columns:
        - unique_id: Track identifier
        - frame: Frame number
        - x_um, y_um: Positions in microns
        - condition: Experimental condition
        - direction_rad (optional): Pre-computed direction
    window_size : int, default=60
        Number of frames per window
    overlap : int, default=30
        Overlap between consecutive windows
    min_track_length : int, default=60
        Minimum track length to consider
        
    Examples
    --------
    >>> dataset = TimeAwareTrajectoryDataset(instant_df, window_size=60, overlap=30)
    >>> print(f"Created {len(dataset)} window pairs")
    >>> 
    >>> # Get a batch
    >>> batch = dataset[0]
    >>> print(batch['features_t'].shape)  # torch.Size([60, 3])
    """
    
    def __init__(self, instant_df, window_size=60, overlap=30, min_track_length=60):
        """
        Args:
            instant_df: Polars or Pandas DataFrame with trajectory data
            window_size: Number of frames per window
            overlap: Overlap between consecutive windows
            min_track_length: Minimum track length (need at least 2 windows)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = window_size - overlap
        self.min_track_length = max(min_track_length, window_size + self.step_size)
        
        # Keep in Polars - much more memory efficient
        if isinstance(instant_df, pl.DataFrame):
            self.df = instant_df.sort(["unique_id", "frame"])
        else:
            self.df = pl.from_pandas(instant_df).sort(["unique_id", "frame"])
        
        # Check if direction_rad column exists
        self.has_direction = 'direction_rad' in self.df.columns
        
        # Step 1: Get track lengths efficiently using Polars aggregation
        print("📊 Analyzing track lengths...")
        track_lengths = self.df.group_by("unique_id").agg([
            pl.len().alias("n_frames"),
            pl.col("condition").first().alias("condition")
        ]).filter(pl.col("n_frames") >= self.min_track_length)
        
        self.valid_tracks = track_lengths.to_dict()
        n_valid = len(self.valid_tracks["unique_id"])
        print(f"   Found {n_valid:,} tracks with >= {self.min_track_length} frames")
        
        # Step 2: Build track lookup with row indices (not data copies!)
        print("📍 Building track index...")
        self._build_track_index_polars()
        
        # Step 3: Generate pair indices
        print("🔗 Generating window pair indices...")
        self.pair_indices = self._generate_pair_indices()
        
        n_pairs = len(self.pair_indices)
        print(f"✅ Created {n_pairs:,} window pair indices")
        print(f"   ⚡ Features computed on-the-fly during training")
    
    def _build_track_index_polars(self):
        """Build index mapping unique_id -> row range in sorted DataFrame"""
        # Add row numbers to sorted dataframe
        df_with_idx = self.df.with_row_index("row_idx")
        
        # Get first and last row index for each track
        track_ranges = df_with_idx.group_by("unique_id").agg([
            pl.col("row_idx").min().alias("start_row"),
            pl.col("row_idx").max().alias("end_row"),
            pl.len().alias("n_frames"),
            pl.col("condition").first().alias("condition"),
        ]).filter(pl.col("n_frames") >= self.min_track_length)
        
        # Convert to dict for fast lookup
        self.track_index = {}
        for row in track_ranges.iter_rows(named=True):
            self.track_index[row["unique_id"]] = {
                "start_row": row["start_row"],
                "end_row": row["end_row"],
                "n_frames": row["n_frames"],
                "condition": row["condition"],
            }
        
        # Pre-extract columns as numpy arrays for fast slicing
        # This is the key optimization - one-time extraction, then numpy slicing
        print("   Extracting coordinate arrays...")
        self.x_um = self.df["x_um"].to_numpy()
        self.y_um = self.df["y_um"].to_numpy()
        self.frames = self.df["frame"].to_numpy()
        if self.has_direction:
            self.direction_rad = self.df["direction_rad"].to_numpy()
        else:
            self.direction_rad = None
    
    def _generate_pair_indices(self):
        """Generate only indices of valid window pairs"""
        pair_indices = []
        
        for unique_id, info in tqdm(self.track_index.items(), desc="Indexing"):
            n_frames = info["n_frames"]
            start_row = info["start_row"]
            
            max_start = n_frames - self.window_size - self.step_size
            
            for local_start in range(0, max_start + 1, self.step_size):
                local_start_t1 = local_start + self.step_size
                local_end_t1 = local_start_t1 + self.window_size
                
                if local_end_t1 <= n_frames:
                    # Store: (unique_id, global_row_start_t, global_row_start_t1)
                    pair_indices.append((
                        unique_id,
                        start_row + local_start,
                        start_row + local_start_t1,
                    ))
        
        return pair_indices
    
    def _extract_features(self, row_start, row_end):
        """Extract motion features from global row indices: [dx, dy, direction]"""
        x = self.x_um[row_start:row_end]
        y = self.y_um[row_start:row_end]
        
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        
        if self.direction_rad is not None:
            direction = self.direction_rad[row_start:row_end]
        else:
            direction = np.arctan2(dy, dx)
            direction = np.nan_to_num(direction, 0)
        
        features = np.column_stack([dx, dy, direction])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features.astype(np.float32)
    
    def __len__(self):
        return len(self.pair_indices)
    
    def __getitem__(self, idx):
        """Generate features ON-THE-FLY using numpy slicing (fast!)"""
        unique_id, row_start_t, row_start_t1 = self.pair_indices[idx]
        
        # Extract features using pre-extracted numpy arrays
        features_t = self._extract_features(row_start_t, row_start_t + self.window_size)
        features_t1 = self._extract_features(row_start_t1, row_start_t1 + self.window_size)
        
        # Get condition from index
        condition = self.track_index[unique_id]["condition"]
        track_start = self.track_index[unique_id]["start_row"]
        
        return {
            'features_t': torch.from_numpy(features_t),
            'features_t1': torch.from_numpy(features_t1),
            'unique_id': unique_id,
            'time_window_t': (row_start_t - track_start) // self.step_size,
            'time_window_t1': (row_start_t1 - track_start) // self.step_size,
            'condition': condition,
        }
    
    def get_single_windows(self):
        """
        Get all unique windows for inference.
        
        Returns a list of dictionaries, each containing:
        - features: numpy array of shape (window_size, 3)
        - unique_id: Track identifier
        - window_uid: Unique window ID (format: unique_id_timewindow_framestart_frameend)
        - time_window: Window index within track
        - start_frame, end_frame: Frame range
        - condition: Experimental condition
        """
        single_windows = []
        seen_keys = set()
        
        for unique_id, row_start_t, row_start_t1 in tqdm(self.pair_indices, desc="Extracting unique windows"):
            info = self.track_index[unique_id]
            track_start = info["start_row"]
            condition = info["condition"]
            
            # Window t
            time_window_t = (row_start_t - track_start) // self.step_size
            start_frame_t = self.frames[row_start_t]
            end_frame_t = self.frames[row_start_t + self.window_size - 1]
            # Create window_uid matching features.py format
            window_uid_t = f"{unique_id}_{time_window_t}_{start_frame_t}_{end_frame_t}"
            key_t = f"{unique_id}_{time_window_t}"
            if key_t not in seen_keys:
                seen_keys.add(key_t)
                single_windows.append({
                    'features': self._extract_features(row_start_t, row_start_t + self.window_size),
                    'unique_id': unique_id,
                    'window_uid': window_uid_t,
                    'time_window': time_window_t,
                    'start_frame': start_frame_t,
                    'end_frame': end_frame_t,
                    'condition': condition,
                })
            
            # Window t+1
            time_window_t1 = (row_start_t1 - track_start) // self.step_size
            start_frame_t1 = self.frames[row_start_t1]
            end_frame_t1 = self.frames[row_start_t1 + self.window_size - 1]
            window_uid_t1 = f"{unique_id}_{time_window_t1}_{start_frame_t1}_{end_frame_t1}"
            key_t1 = f"{unique_id}_{time_window_t1}"
            if key_t1 not in seen_keys:
                seen_keys.add(key_t1)
                single_windows.append({
                    'features': self._extract_features(row_start_t1, row_start_t1 + self.window_size),
                    'unique_id': unique_id,
                    'window_uid': window_uid_t1,
                    'time_window': time_window_t1,
                    'start_frame': start_frame_t1,
                    'end_frame': end_frame_t1,
                    'condition': condition,
                })
        
        return single_windows


def contrastive_loss(z_i, z_j, temperature=0.5):
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(
        representations.unsqueeze(1), representations.unsqueeze(0), dim=2
    )

    N = z_i.shape[0]
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0)
    labels = labels.to(z_i.device)

    mask = ~torch.eye(2 * N, dtype=bool).to(z_i.device)
    logits = similarity_matrix[mask].view(2 * N, -1)
    positives = torch.exp(similarity_matrix[torch.arange(2 * N), labels] / temperature)
    denominator = torch.sum(torch.exp(logits / temperature), dim=1)
    loss = -torch.log(positives / denominator)
    return loss.mean()


def time_aware_contrastive_loss(z_t, z_t1, temperature=0.5, track_ids=None, mask_same_track=False):
    """
    Time-aware contrastive loss with optional same-track masking.
    
    Designed for temporal contrastive learning where positive pairs are 
    temporally adjacent windows from the same track.
    
    Parameters
    ----------
    z_t : torch.Tensor
        Embeddings for windows at time t, shape [B, embed_dim]
    z_t1 : torch.Tensor
        Embeddings for windows at time t+1, shape [B, embed_dim]
    temperature : float, default=0.5
        Temperature for softmax scaling. Lower = sharper distribution
    track_ids : list, optional
        List of track IDs for each sample (needed if mask_same_track=True)
    mask_same_track : bool, default=False
        If True, exclude same-track pairs from negatives
    
    Returns
    -------
    torch.Tensor
        Scalar contrastive loss value
        
    Notes
    -----
    Positive pairs: (z_t[i], z_t1[i]) - temporally adjacent windows from same track
    Negative pairs: All other combinations (optionally excluding same-track pairs)
    
    Examples
    --------
    >>> loss = time_aware_contrastive_loss(z_t, z_t1, temperature=0.5)
    >>> loss = time_aware_contrastive_loss(z_t, z_t1, track_ids=track_ids, mask_same_track=True)
    """
    # Normalize embeddings
    z_t = F.normalize(z_t, dim=1)
    z_t1 = F.normalize(z_t1, dim=1)
    
    batch_size = z_t.shape[0]
    device = z_t.device
    
    # Concatenate all embeddings: [z_t; z_t1] -> [2B, embed_dim]
    representations = torch.cat([z_t, z_t1], dim=0)
    
    # Compute similarity matrix [2B, 2B]
    similarity_matrix = F.cosine_similarity(
        representations.unsqueeze(1), 
        representations.unsqueeze(0), 
        dim=2
    )
    
    # Create labels: positive pairs are (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(batch_size) + batch_size,
        torch.arange(batch_size)
    ]).to(device)
    
    # Base mask: exclude self-similarity (diagonal)
    mask = ~torch.eye(2 * batch_size, dtype=bool, device=device)
    
    # Optional: also exclude same-track pairs from negatives
    if mask_same_track and track_ids is not None:
        # Build same-track mask
        # track_ids is [B] - duplicate for [2B]
        track_ids_2b = track_ids + track_ids  # Python list concatenation
        
        # same_track[i,j] = True if track_ids_2b[i] == track_ids_2b[j]
        same_track = torch.zeros(2 * batch_size, 2 * batch_size, dtype=bool, device=device)
        for i in range(2 * batch_size):
            for j in range(2 * batch_size):
                if track_ids_2b[i] == track_ids_2b[j]:
                    same_track[i, j] = True
        
        # Exclude same-track pairs (but keep positives!)
        # We want to mask same-track NON-positives
        positive_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=bool, device=device)
        positive_mask[torch.arange(2 * batch_size), labels] = True
        
        # Final mask: not self, not same-track (unless it's the positive)
        mask = mask & (~same_track | positive_mask)
    
    # Get positive similarities
    positives = similarity_matrix[torch.arange(2 * batch_size), labels]
    
    # Apply temperature scaling to similarity matrix
    sim_scaled = similarity_matrix / temperature
    
    # Set masked positions to -inf (so exp(-inf) = 0)
    sim_scaled = sim_scaled.masked_fill(~mask, float('-inf'))
    
    # Compute loss using logsumexp for numerical stability
    # loss = -log(exp(pos/T) / sum(exp(neg/T)))
    # = -pos/T + log(sum(exp(neg/T)))
    log_sum_exp = torch.logsumexp(sim_scaled, dim=1)
    loss = -positives / temperature + log_sum_exp
    
    return loss.mean()


def combined_contrastive_loss(z_t, z_t1, z_t_aug, temperature=0.5, temporal_weight=0.5,
                               track_ids=None, mask_same_track=False):
    """
    Combined loss: temporal adjacency + augmentation invariance.
    
    Combines two contrastive objectives:
    1. Temporal: adjacent windows should have similar embeddings
    2. Augmentation: original and augmented views should match
    
    Parameters
    ----------
    z_t : torch.Tensor
        Embeddings for windows at time t
    z_t1 : torch.Tensor
        Embeddings for windows at time t+1 (temporal positive)
    z_t_aug : torch.Tensor
        Embeddings for augmented windows at time t (augmentation positive)
    temperature : float, default=0.5
        Softmax temperature
    temporal_weight : float, default=0.5
        Weight for temporal loss (1 - temporal_weight for augmentation loss)
    track_ids : list, optional
        Track IDs for same-track masking
    mask_same_track : bool, default=False
        Whether to mask same-track negatives
    
    Returns
    -------
    tuple
        (combined_loss, temporal_loss, augmentation_loss)
        
    Examples
    --------
    >>> combined, temp_loss, aug_loss = combined_contrastive_loss(z_t, z_t1, z_t_aug)
    >>> print(f"Temporal: {temp_loss:.4f}, Augmentation: {aug_loss:.4f}")
    """
    # Temporal loss
    temporal_loss = time_aware_contrastive_loss(
        z_t, z_t1, temperature, track_ids, mask_same_track
    )
    
    # Augmentation loss
    augmentation_loss = time_aware_contrastive_loss(
        z_t, z_t_aug, temperature, track_ids, mask_same_track
    )
    
    combined = temporal_weight * temporal_loss + (1 - temporal_weight) * augmentation_loss
    
    return combined, temporal_loss, augmentation_loss


def within_window_contrastive_loss(z_even, z_odd, temperature=0.5, track_ids=None, mask_same_track=False):
    """
    Within-window contrastive loss using interleaved frame views.
    
    Creates positive pairs from non-overlapping subsets of the SAME window:
    - View A: even-indexed frames [0, 2, 4, ..., 58]
    - View B: odd-indexed frames [1, 3, 5, ..., 59]
    
    This enforces within-window consistency: "all parts of this 600ms window
    should produce the same embedding" without assuming anything about
    relationships between different windows.
    
    Parameters
    ----------
    z_even : torch.Tensor
        Embeddings from even-indexed frames, shape [B, embed_dim]
    z_odd : torch.Tensor
        Embeddings from odd-indexed frames, shape [B, embed_dim]
    temperature : float, default=0.5
        Temperature for softmax scaling
    track_ids : list, optional
        List of track IDs for same-track masking
    mask_same_track : bool, default=False
        If True, exclude same-track pairs from negatives
    
    Returns
    -------
    torch.Tensor
        Scalar contrastive loss value
        
    Notes
    -----
    Positive pairs: (z_even[i], z_odd[i]) - different views of SAME window
    Negative pairs: Views from DIFFERENT windows (optionally excluding same-track)
    
    The key difference from temporal contrastive loss:
    - Temporal: (window_t, window_{t+1}) share 50% frames - too easy!
    - Within-window: (even_frames, odd_frames) share 0% frames - actually tests consistency
    
    Examples
    --------
    >>> # Split window into interleaved views
    >>> even_frames = window[:, 0::2, :]  # [B, 30, 3]
    >>> odd_frames = window[:, 1::2, :]   # [B, 30, 3]
    >>> z_even = model(even_frames)
    >>> z_odd = model(odd_frames)
    >>> loss = within_window_contrastive_loss(z_even, z_odd, temperature=0.5)
    """
    # Use the same loss structure as time_aware_contrastive_loss
    # but with a clearer semantic meaning
    return time_aware_contrastive_loss(
        z_even, z_odd, temperature, track_ids, mask_same_track
    )


def adjacent_subwindow_contrastive_loss(
    subwindow_embeddings, 
    track_ids,
    temperature=0.7,
    mask_same_track=True,
):
    """
    Adjacent subwindow contrastive loss for temporal locality learning.
    
    Divides each window into non-overlapping subwindows and creates positive
    pairs from ADJACENT subwindows (no shared frames). This teaches the model
    that "temporally close = behaviorally similar" without the shortcut of
    matching on identical frames.
    
    Parameters
    ----------
    subwindow_embeddings : torch.Tensor
        Shape [B, num_subwindows, embed_dim] where:
        - B = batch size (number of full windows)
        - num_subwindows = 6 (for 60-frame windows with 10-frame subwindows)
        - embed_dim = embedding dimension
    track_ids : list
        List of track IDs for each window in the batch (length B).
        Used to mask same-track negatives.
    temperature : float, default=0.7
        Temperature for softmax scaling. Higher = softer distinctions.
        Use 0.7 (softer) to be tolerant of similar behaviors in negatives.
    mask_same_track : bool, default=True
        If True, exclude same-track subwindows (non-adjacent) from negatives.
        
    Returns
    -------
    torch.Tensor
        Scalar contrastive loss value
        
    Notes
    -----
    For a batch of B windows, each with 6 subwindows:
    
    Positive pairs (5 per window, 5B total):
        (sub_i_0, sub_i_1), (sub_i_1, sub_i_2), ..., (sub_i_4, sub_i_5)
        
    Negatives for anchor sub_i_k:
        All subwindows from OTHER windows: {sub_j_m} for j ≠ i
        
    Masked (neither positive nor negative):
        Non-adjacent subwindows from SAME window: sub_i_0 and sub_i_3
        
    This structure:
    - Forces learning temporal locality (adjacent = similar)
    - Avoids trivial matching on shared frames (0 overlap)
    - Uses soft temperature to tolerate similar behaviors across tracks
    
    Examples
    --------
    >>> # Extract subwindows from 60-frame windows
    >>> B, T, F = 64, 60, 3
    >>> windows = batch['features']  # [64, 60, 3]
    >>> subwindows = windows.reshape(B, 6, 10, F)  # [64, 6, 10, 3]
    >>> 
    >>> # Get embeddings for each subwindow
    >>> z_subs = model(subwindows.reshape(B*6, 10, F))  # [384, embed_dim]
    >>> z_subs = z_subs.reshape(B, 6, -1)  # [64, 6, embed_dim]
    >>> 
    >>> # Compute loss
    >>> loss = adjacent_subwindow_contrastive_loss(z_subs, track_ids, temperature=0.7)
    """
    device = subwindow_embeddings.device
    B, num_subs, embed_dim = subwindow_embeddings.shape
    
    # Number of adjacent pairs per window
    num_pairs = num_subs - 1  # 5 for 6 subwindows
    
    # Create pairs: anchor (sub_k) and positive (sub_{k+1})
    # Shape: [B * num_pairs, embed_dim]
    anchors = []
    positives = []
    pair_track_ids = []
    pair_window_ids = []  # To track which window each pair came from
    
    for k in range(num_pairs):
        anchors.append(subwindow_embeddings[:, k, :])      # [B, embed_dim]
        positives.append(subwindow_embeddings[:, k+1, :])  # [B, embed_dim]
        pair_track_ids.extend(track_ids)
        pair_window_ids.extend(range(B))
    
    z_anchor = torch.cat(anchors, dim=0)    # [B * num_pairs, embed_dim]
    z_positive = torch.cat(positives, dim=0)  # [B * num_pairs, embed_dim]
    
    # Normalize embeddings
    z_anchor = F.normalize(z_anchor, dim=1)
    z_positive = F.normalize(z_positive, dim=1)
    
    # Also normalize all subwindow embeddings for negative computation
    all_z = F.normalize(subwindow_embeddings.reshape(B * num_subs, embed_dim), dim=1)  # [B*6, embed_dim]
    
    N = z_anchor.shape[0]  # Total number of positive pairs (B * num_pairs)
    
    # Positive similarities: each anchor with its adjacent positive
    pos_sim = torch.sum(z_anchor * z_positive, dim=1) / temperature  # [N]
    
    # Negative similarities: each anchor with ALL subwindows from OTHER windows
    # Compute all pairwise similarities: [N, B*num_subs]
    all_sim = torch.mm(z_anchor, all_z.t()) / temperature  # [N, B*6]
    
    # Create mask for negatives
    # We need to mask out:
    # 1. The positive pair itself
    # 2. All subwindows from the SAME WINDOW (not just same track)
    # 3. Optionally, all subwindows from the same TRACK (if mask_same_track=True)
    
    mask = torch.ones(N, B * num_subs, dtype=torch.bool, device=device)
    
    for pair_idx in range(N):
        window_idx = pair_window_ids[pair_idx]
        
        # Mask all subwindows from the same window (neither positive nor negative)
        for sub_idx in range(num_subs):
            mask[pair_idx, window_idx * num_subs + sub_idx] = False
    
    # Additionally mask same-track windows (different window, same track)
    if mask_same_track and track_ids is not None:
        for pair_idx in range(N):
            pair_track = pair_track_ids[pair_idx]
            for other_window in range(B):
                if track_ids[other_window] == pair_track:
                    # Mask all subwindows from this same-track window
                    for sub_idx in range(num_subs):
                        mask[pair_idx, other_window * num_subs + sub_idx] = False
    
    # Apply mask: set masked positions to very negative value
    all_sim = all_sim.masked_fill(~mask, -1e9)
    
    # Concatenate positive and negatives for softmax
    # logits shape: [N, 1 + B*num_subs] where first column is positive
    logits = torch.cat([pos_sim.unsqueeze(1), all_sim], dim=1)  # [N, 1 + B*6]
    
    # Labels: positive is always at index 0
    labels = torch.zeros(N, dtype=torch.long, device=device)
    
    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    
    return loss


def extract_subwindows(features, subwindow_size=10):
    """
    Extract non-overlapping subwindows from full windows.
    
    Parameters
    ----------
    features : torch.Tensor
        Shape [B, T, F] - batch of windows
    subwindow_size : int, default=10
        Size of each subwindow in frames
        
    Returns
    -------
    torch.Tensor
        Shape [B, num_subwindows, subwindow_size, F]
        
    Examples
    --------
    >>> features = torch.randn(64, 60, 3)  # [B, T, F]
    >>> subwindows = extract_subwindows(features, subwindow_size=10)
    >>> print(subwindows.shape)  # [64, 6, 10, 3]
    """
    B, T, F = features.shape
    num_subwindows = T // subwindow_size
    
    # Reshape to extract non-overlapping subwindows
    # [B, T, F] -> [B, num_subwindows, subwindow_size, F]
    subwindows = features[:, :num_subwindows * subwindow_size, :].reshape(
        B, num_subwindows, subwindow_size, F
    )
    
    return subwindows


# =============================================================================
# EVALUATION FUNCTIONS: Track Identity vs Behavior Clustering
# =============================================================================

def evaluate_track_identity_leakage(embeddings, track_ids, n_samples=1000, random_seed=42):
    """
    Check if embeddings cluster by track identity (BAD) or by behavior (GOOD).
    
    Computes the ratio of intra-track similarity to inter-track similarity.
    A ratio close to 1.0 means the model learned behavior, not track identity.
    A ratio >> 1 means the model learned to identify tracks (overfitting).
    
    Parameters
    ----------
    embeddings : np.ndarray
        Shape [N, embed_dim] - embeddings from windows
    track_ids : np.ndarray or list
        Track IDs corresponding to each embedding
    n_samples : int, default=1000
        Number of pairs to sample for each comparison
    random_seed : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    dict
        {
            'intra_track_sim': float,  # Mean similarity within tracks
            'inter_track_sim': float,  # Mean similarity between tracks
            'ratio': float,            # intra / inter (closer to 1.0 = better)
            'verdict': str,            # 'GOOD', 'WARNING', or 'BAD'
        }
        
    Examples
    --------
    >>> result = evaluate_track_identity_leakage(embeddings, track_ids)
    >>> print(f"Ratio: {result['ratio']:.2f} - {result['verdict']}")
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    np.random.seed(random_seed)
    
    embeddings = np.array(embeddings)
    track_ids = np.array(track_ids)
    
    unique_tracks = np.unique(track_ids)
    
    # Compute intra-track similarities (same track, different windows)
    intra_sims = []
    for track in unique_tracks:
        track_mask = track_ids == track
        track_embeds = embeddings[track_mask]
        
        if len(track_embeds) > 1:
            # Sample pairs within this track
            n_pairs = min(len(track_embeds) * (len(track_embeds) - 1) // 2, n_samples // len(unique_tracks))
            for _ in range(max(1, n_pairs)):
                i, j = np.random.choice(len(track_embeds), size=2, replace=False)
                sim = cosine_similarity(track_embeds[i:i+1], track_embeds[j:j+1])[0, 0]
                intra_sims.append(sim)
    
    # Compute inter-track similarities (different tracks)
    inter_sims = []
    for _ in range(n_samples):
        # Pick two different tracks
        t1, t2 = np.random.choice(unique_tracks, size=2, replace=False)
        
        # Pick one embedding from each
        e1 = embeddings[track_ids == t1][np.random.randint(sum(track_ids == t1))]
        e2 = embeddings[track_ids == t2][np.random.randint(sum(track_ids == t2))]
        
        sim = cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0, 0]
        inter_sims.append(sim)
    
    intra_mean = np.mean(intra_sims) if intra_sims else 0.0
    inter_mean = np.mean(inter_sims) if inter_sims else 0.0
    
    ratio = intra_mean / inter_mean if inter_mean != 0 else float('inf')
    
    # Verdict
    if ratio < 1.2:
        verdict = "GOOD - Model learned behavior, not track identity"
    elif ratio < 1.5:
        verdict = "WARNING - Some track identity leakage detected"
    else:
        verdict = "BAD - Model learned track identity, not behavior"
    
    return {
        'intra_track_sim': intra_mean,
        'inter_track_sim': inter_mean,
        'ratio': ratio,
        'verdict': verdict,
    }


def evaluate_behavior_clustering(embeddings, cluster_labels, motion_features, feature_names=None):
    """
    Check if embedding clusters correspond to distinct behavioral states.
    
    Uses ANOVA to test if clusters have significantly different motion features.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Shape [N, embed_dim]
    cluster_labels : np.ndarray
        Cluster assignments for each embedding
    motion_features : np.ndarray or pd.DataFrame
        Motion features for each window (e.g., speed, alpha, confinement)
    feature_names : list, optional
        Names of motion features (columns of motion_features)
        
    Returns
    -------
    dict
        {
            'feature_anova': dict,      # {feature: {'F': f_stat, 'p': p_val}}
            'silhouette_motion': float, # Silhouette score on motion features
            'verdict': str,
        }
        
    Examples
    --------
    >>> motion_feats = windowed_df[['avg_speed_um_s', 'anomalous_exponent']].to_numpy()
    >>> result = evaluate_behavior_clustering(embeddings, clusters, motion_feats)
    >>> print(f"Silhouette: {result['silhouette_motion']:.3f}")
    """
    from scipy.stats import f_oneway
    from sklearn.metrics import silhouette_score
    
    embeddings = np.array(embeddings)
    cluster_labels = np.array(cluster_labels)
    
    if hasattr(motion_features, 'to_numpy'):
        motion_features = motion_features.to_numpy()
    motion_features = np.array(motion_features)
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(motion_features.shape[1])]
    
    unique_clusters = np.unique(cluster_labels)
    
    # ANOVA for each feature
    feature_anova = {}
    significant_count = 0
    
    for i, feat_name in enumerate(feature_names):
        feat_values = motion_features[:, i]
        groups = [feat_values[cluster_labels == c] for c in unique_clusters]
        
        # Remove empty groups
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            try:
                f_stat, p_val = f_oneway(*groups)
                feature_anova[feat_name] = {'F': f_stat, 'p': p_val}
                if p_val < 0.01:
                    significant_count += 1
            except:
                feature_anova[feat_name] = {'F': 0.0, 'p': 1.0}
    
    # Silhouette score on motion features (do clusters separate in motion space?)
    try:
        sil_motion = silhouette_score(motion_features, cluster_labels)
    except:
        sil_motion = 0.0
    
    # Verdict
    sig_frac = significant_count / len(feature_names) if feature_names else 0
    if sil_motion > 0.2 and sig_frac > 0.5:
        verdict = "GOOD - Clusters represent distinct behavioral states"
    elif sil_motion > 0.1 or sig_frac > 0.3:
        verdict = "MODERATE - Some behavioral structure detected"
    else:
        verdict = "POOR - Clusters don't correspond to clear behaviors"
    
    return {
        'feature_anova': feature_anova,
        'silhouette_motion': sil_motion,
        'verdict': verdict,
    }


def evaluate_cross_molecule_clustering(cluster_labels, molecule_types, n_clusters=None):
    """
    Check if clusters contain diverse molecules (GOOD) or are dominated by one (BAD).
    
    If the model learned behavior (not molecule identity), clusters should
    contain multiple molecule types.
    
    Parameters
    ----------
    cluster_labels : np.ndarray
        Cluster assignments
    molecule_types : np.ndarray or list
        Molecule type for each window
    n_clusters : int, optional
        Number of clusters (inferred if not provided)
        
    Returns
    -------
    dict
        {
            'cluster_diversity': dict,  # {cluster: {'n_molecules': N, 'entropy': H, 'dominant': mol}}
            'mean_entropy': float,
            'verdict': str,
        }
    """
    from scipy.stats import entropy
    from collections import Counter
    
    cluster_labels = np.array(cluster_labels)
    molecule_types = np.array(molecule_types)
    
    unique_clusters = np.unique(cluster_labels)
    
    cluster_diversity = {}
    entropies = []
    
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        mols_in_cluster = molecule_types[mask]
        mol_counts = Counter(mols_in_cluster)
        
        n_molecules = len(mol_counts)
        counts = np.array(list(mol_counts.values()))
        probs = counts / counts.sum()
        h = entropy(probs)  # Shannon entropy
        dominant = mol_counts.most_common(1)[0][0] if mol_counts else None
        dominant_frac = mol_counts.most_common(1)[0][1] / len(mols_in_cluster) if mol_counts else 0
        
        cluster_diversity[int(cluster)] = {
            'n_molecules': n_molecules,
            'entropy': h,
            'dominant': dominant,
            'dominant_frac': dominant_frac,
        }
        entropies.append(h)
    
    mean_entropy = np.mean(entropies)
    
    # Verdict
    if mean_entropy > 1.0:
        verdict = "GOOD - Clusters contain diverse molecules (learned behavior)"
    elif mean_entropy > 0.5:
        verdict = "MODERATE - Some molecule diversity in clusters"
    else:
        verdict = "WARNING - Clusters dominated by single molecules (may have learned identity)"
    
    return {
        'cluster_diversity': cluster_diversity,
        'mean_entropy': mean_entropy,
        'verdict': verdict,
    }


# =============================================================================
# AUGMENTATION FUNCTIONS FOR CONTRASTIVE LEARNING
# =============================================================================

def get_augmentation_fn(aug_type='noise', strength=0.01, noise_strength=None, scale_strength=None):
    """
    Get augmentation function for contrastive learning.
    
    Returns a function that applies the specified augmentation to input tensors.
    Used to create positive pairs in contrastive learning by generating
    augmented views of the same data.
    
    Parameters
    ----------
    aug_type : str, default='noise'
        Augmentation type: 'noise', 'time_warp', 'scale', 'crop', 'combined'
    strength : float, default=0.01
        Augmentation strength (meaning varies by type)
    noise_strength : float, optional
        Separate noise strength for 'combined' type
    scale_strength : float, optional
        Separate scale strength for 'combined' type
    
    Returns
    -------
    callable
        Augmentation function: x -> augmented_x
        
    Examples
    --------
    >>> aug_fn = get_augmentation_fn('noise', strength=0.01)
    >>> x_augmented = aug_fn(x)
    
    >>> aug_fn = get_augmentation_fn('combined', noise_strength=0.01, scale_strength=0.1)
    >>> x_augmented = aug_fn(x)
    
    Notes
    -----
    Augmentation types:
    - 'noise': Gaussian noise added to features
    - 'scale': Random scaling of dx, dy (first 2 features)
    - 'time_warp': Simplified time warping via noise
    - 'crop': Zero out start or end of sequence
    - 'combined': Apply both noise and scale
    """
    
    if aug_type == 'noise':
        def noise_aug(x):
            """Add Gaussian noise to motion features"""
            return x + strength * torch.randn_like(x)
        return noise_aug
    
    elif aug_type == 'scale':
        def scale_aug(x):
            """Random scaling of displacement magnitudes"""
            # Scale factor between (1-strength) and (1+strength)
            scale = 1.0 + (torch.rand(1, device=x.device) * 2 - 1) * strength
            # Only scale dx, dy (first 2 features), not direction
            x_aug = x.clone()
            x_aug[:, :, :2] = x[:, :, :2] * scale
            return x_aug
        return scale_aug
    
    elif aug_type == 'time_warp':
        def time_warp_aug(x):
            """Slight time warping by interpolation"""
            # This is a simplified version - full DTW-based warping is complex
            # Here we just add small random shifts to the sequence
            B, T, F = x.shape
            noise = torch.randn(B, T, 1, device=x.device) * strength * 0.1
            # Interpolate features based on noise (simplified)
            return x + noise * x
        return time_warp_aug
    
    elif aug_type == 'crop':
        def crop_aug(x):
            """Random cropping and padding"""
            B, T, F = x.shape
            crop_amount = int(T * strength * 0.5)
            if crop_amount < 1:
                return x
            # Random crop from start or end
            x_aug = x.clone()
            if torch.rand(1) > 0.5:
                x_aug[:, :crop_amount, :] = 0  # Zero out start
            else:
                x_aug[:, -crop_amount:, :] = 0  # Zero out end
            return x_aug
        return crop_aug
    
    elif aug_type == 'combined':
        # Use separate strengths if provided, otherwise fall back to single strength
        ns = noise_strength if noise_strength is not None else strength
        ss = scale_strength if scale_strength is not None else strength
        noise_fn = get_augmentation_fn('noise', ns)
        scale_fn = get_augmentation_fn('scale', ss)
        def combined_aug(x):
            """Apply multiple augmentations"""
            x = noise_fn(x)
            x = scale_fn(x)
            return x
        return combined_aug
    
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


class MotionTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        lr=1e-4,
        device="cuda",
        use_tensorboard=False,
        tensorboard_log_dir=None,
        augmentation_strategy="basic",
        use_scheduler=False,
    ):
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
                self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
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
                print(
                    "   ⚠ TensorBoard not available. Install with: mamba install tensorboard"
                )
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
                batch_features = batch["features"].to(self.device)
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
                    batch_features = batch["features"].to(self.device)
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
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                if val_loss is not None:
                    self.writer.add_scalar("Loss/Validation", val_loss, epoch)
                self.writer.add_scalar(
                    "Learning_Rate", self.optimizer.param_groups[0]["lr"], epoch
                )
                self.writer.flush()

            # Print progress
            current_lr = self.optimizer.param_groups[0]["lr"]
            if val_loss is not None:
                print(
                    f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.2e}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, LR = {current_lr:.2e}"
                )

        # Close TensorBoard writer
        if self.use_tensorboard and self.writer:
            self.writer.close()

    def plot_loss_curve(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(12, 5))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Training Loss", color="blue")
        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)

        # Plot learning rate if scheduler is used
        if self.use_scheduler and len(self.train_losses) > 1:
            plt.subplot(1, 2, 2)
            # This is approximate - in practice you'd store LR history
            plt.plot(
                range(len(self.train_losses)),
                [self.optimizer.param_groups[0]["lr"]] * len(self.train_losses),
            )
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
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
                    batch_features = batch["features"].to(self.device)
                    # Extract metadata for each item in the batch
                    batch_metadata = []
                    for i in range(len(batch_features)):
                        metadata = {}
                        for key, value in batch.items():
                            if key != "features":  # Skip the features tensor
                                if isinstance(value, list):
                                    metadata[key] = value[i] if i < len(value) else None
                                elif hasattr(value, "__getitem__"):
                                    try:
                                        metadata[key] = (
                                            value[i].item()
                                            if hasattr(value[i], "item")
                                            else value[i]
                                        )
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
                    batch_features = batch["features"].to(self.device)
                    # Extract metadata for each item in the batch
                    for i in range(len(batch_features)):
                        metadata = {"batch_idx": batch_idx, "within_batch_idx": i}
                        for key, value in batch.items():
                            if key != "features":  # Skip the features tensor
                                if isinstance(value, list):
                                    metadata[key] = value[i] if i < len(value) else None
                                elif hasattr(value, "__getitem__"):
                                    try:
                                        metadata[key] = (
                                            value[i].item()
                                            if hasattr(value[i], "item")
                                            else value[i]
                                        )
                                    except:
                                        metadata[key] = None
                                else:
                                    metadata[key] = value
                        all_metadata.append(metadata)
                else:
                    batch_features = batch.to(self.device)
                    # No metadata available for old format
                    for i in range(len(batch_features)):
                        all_metadata.append(
                            {"batch_idx": batch_idx, "within_batch_idx": i}
                        )

                embeddings = self.model(batch_features)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings), all_metadata

    def cluster_embeddings(self, embeddings, n_clusters=5):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)

    def save_model(self, path):
        """Save model state dict and training info"""
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,  # NEW: Save validation losses
            "augmentation_strategy": self.augmentation_strategy,
            "device": str(self.device),
        }
        torch.save(save_dict, path)
        print(f"✅ Model saved to: {path}")

    def load_model(self, path, load_optimizer=True):
        """Load model state dict and optionally optimizer state"""
        checkpoint = torch.load(path, map_location=self.device)

        # Load model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        # Load optimizer if requested and available
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load training history
        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
        elif "losses" in checkpoint:  # Backward compatibility
            self.train_losses = checkpoint["losses"]

        if "val_losses" in checkpoint:
            self.val_losses = checkpoint["val_losses"]

        # Update losses reference for backward compatibility
        self.losses = self.train_losses

        # Load augmentation strategy
        if "augmentation_strategy" in checkpoint:
            self.augmentation_strategy = checkpoint["augmentation_strategy"]
            self._setup_augmentation()

        print(f"✅ Model loaded from: {path}")
        print(f"   Previous training epochs: {len(self.train_losses)}")
        if self.val_losses:
            print(f"   Previous validation epochs: {len(self.val_losses)}")
        if "augmentation_strategy" in checkpoint:
            print(f"   Augmentation strategy: {checkpoint['augmentation_strategy']}")

    def resume_training(self, epochs=10):
        """Resume training from current state"""
        print(f"🔄 Resuming training from epoch {len(self.train_losses) + 1}")
        self.train(epochs)


class TimeAwareMotionTrainer:
    """
    Trainer for contrastive learning on motion data with multiple learning objectives.
    
    Supports three contrastive learning objectives that can be combined:
    1. Augmentation contrastive loss (FOUNDATION): (window, augmented_window)
    2. Within-window consistency (RECOMMENDED): (even_frames, odd_frames) of same window
    3. Temporal contrastive loss (LEGACY - often too easy with overlapping windows)
    
    The within-window consistency loss is the key innovation: it creates positive
    pairs from non-overlapping subsets of the SAME window (interleaved frames),
    enforcing that all parts of a 600ms window represent the same behavioral state.
    
    Features:
    1. Augmentation contrastive loss (same window ± augmentation) - always on
    2. Within-window consistency loss (even vs odd frames) - toggleable
    3. Temporal contrastive loss (adjacent windows) - legacy, often too easy
    4. Same-track negative masking (prevents same-track windows as negatives)
    5. Multiple augmentation types
    6. TensorBoard logging
    7. Learning rate scheduling
    8. Best model saving (based on validation loss)
    9. Checkpoint saving at configurable intervals
    
    Parameters
    ----------
    model : nn.Module
        The encoder model (e.g., TransformerMotionEncoder)
    train_dataloader : DataLoader
        DataLoader yielding TimeAwareTrajectoryDataset batches
    val_dataloader : DataLoader, optional
        Validation DataLoader
    lr : float, default=1e-4
        Learning rate
    device : str, default="cuda"
        Device to train on
    temperature : float, default=0.5
        Temperature for contrastive loss
    temporal_weight : float, default=0.0
        Weight for temporal loss. Set to 0 to disable (RECOMMENDED).
        The temporal loss with overlapping windows is often too easy.
    use_augmentation : bool, default=True
        Whether to use augmentation contrastive loss (ALWAYS RECOMMENDED)
    use_within_window_consistency : bool, default=False
        Whether to use within-window consistency loss (interleaved frame views).
        This enforces that different subsets of the same 600ms window produce
        similar embeddings, without assuming anything about adjacent windows.
    within_window_weight : float, default=0.5
        Weight for within-window consistency loss (if enabled)
    use_scheduler : bool, default=True
        Whether to use learning rate scheduler
    use_tensorboard : bool, default=False
        Whether to log to TensorBoard
    tensorboard_log_dir : str, optional
        Directory for TensorBoard logs
    mask_same_track : bool, default=True
        Whether to mask same-track negatives. RECOMMENDED to keep True.
    augmentation_type : str, default='noise'
        Type of augmentation: 'noise', 'scale', 'combined', etc.
    augmentation_strength : float, default=0.01
        Strength of augmentation
    noise_strength : float, optional
        Separate noise strength for 'combined' augmentation
    scale_strength : float, optional
        Separate scale strength for 'combined' augmentation
    augment_temporal_pairs : bool, default=False
        Whether to also augment the temporal window pairs
    save_best_model : bool, default=True
        Whether to track and save the model state with the lowest validation loss.
        After training, call restore_best_model() to restore to the optimal state.
    checkpoint_interval : int, default=0
        Save checkpoints every N epochs. Set to 0 to disable checkpoint saving.
        Checkpoints are saved to checkpoint_dir with names like 'checkpoint_epoch_10.pt'
    checkpoint_dir : str, optional
        Directory to save checkpoints. If None, uses tensorboard_log_dir or current dir.
        
    Examples
    --------
    >>> # RECOMMENDED: Augmentation + Within-window consistency, no temporal
    >>> trainer = TimeAwareMotionTrainer(
    ...     model=encoder,
    ...     train_dataloader=train_loader,
    ...     temporal_weight=0.0,  # Disable temporal (too easy with overlap)
    ...     use_within_window_consistency=True,  # Enable within-window
    ...     within_window_weight=0.5,
    ...     mask_same_track=True,
    ...     augmentation_type='combined',
    ...     save_best_model=True,  # Track best validation loss
    ...     checkpoint_interval=10,  # Save every 10 epochs
    ... )
    
    >>> # After training, restore best model
    >>> trainer.train(epochs=100)
    >>> trainer.restore_best_model()  # Go back to best validation epoch
    
    >>> # Pure augmentation only (simplest, still good)
    >>> trainer = TimeAwareMotionTrainer(
    ...     model=encoder,
    ...     train_dataloader=train_loader,
    ...     temporal_weight=0.0,
    ...     use_within_window_consistency=False,
    ...     mask_same_track=True,
    ... )
    """
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        lr=1e-4,
        device="cuda",
        temperature=0.5,
        temporal_weight=0.5,
        use_augmentation=True,
        use_within_window_consistency=False,
        within_window_weight=0.5,
        use_scheduler=True,
        use_tensorboard=False,
        tensorboard_log_dir=None,
        # Contrastive learning options
        mask_same_track=False,
        augmentation_type='noise',
        augmentation_strength=0.01,
        noise_strength=None,
        scale_strength=None,
        augment_temporal_pairs=False,
        # Adjacent subwindow contrastive (NEW - RECOMMENDED)
        use_adjacent_subwindow=False,
        adjacent_subwindow_weight=0.5,
        adjacent_temperature=0.7,  # Softer temperature for subwindow loss
        subwindow_size=10,
        # Checkpoint and best model options
        save_best_model=True,
        checkpoint_interval=0,
        checkpoint_dir=None,
        # Early stopping
        early_stopping_patience=0,  # 0 = disabled, N = stop after N epochs without improvement
        # Resume support
        epoch_offset=0,  # Starting epoch (for resume from checkpoint)
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.temperature = temperature
        self.temporal_weight = temporal_weight
        self.use_augmentation = use_augmentation
        self.use_within_window_consistency = use_within_window_consistency
        self.within_window_weight = within_window_weight
        
        # Adjacent subwindow contrastive (NEW)
        self.use_adjacent_subwindow = use_adjacent_subwindow
        self.adjacent_subwindow_weight = adjacent_subwindow_weight
        self.adjacent_temperature = adjacent_temperature
        self.subwindow_size = subwindow_size
        
        # Contrastive learning options
        self.mask_same_track = mask_same_track
        self.augment_temporal_pairs = augment_temporal_pairs
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Learning rate scheduler
        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self.tensorboard_log_dir = tensorboard_log_dir
        self.writer = None
        if use_tensorboard and tensorboard_log_dir:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
            print(f"📊 TensorBoard logging to: {tensorboard_log_dir}")
        
        # Loss history - Training
        self.train_losses = []
        self.temporal_losses = []
        self.augmentation_losses = []
        self.within_window_losses = []
        self.adjacent_subwindow_losses = []  # NEW
        
        # Loss history - Validation (matching training components)
        self.val_losses = []
        self.val_temporal_losses = []
        self.val_augmentation_losses = []
        self.val_within_window_losses = []
        self.val_adjacent_subwindow_losses = []  # NEW
        
        # Best model tracking (for early stopping / optimal epoch selection)
        self.save_best_model = save_best_model
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_epoch = -1
        
        # Checkpoint saving
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir or tensorboard_log_dir or "."
        if checkpoint_interval > 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0
        
        # Resume support - epoch offset for correct TensorBoard/checkpoint numbering
        self.epoch_offset = epoch_offset
        
        # Use configurable augmentation with separate strengths for 'combined'
        self.augmentation_fn = get_augmentation_fn(
            augmentation_type, augmentation_strength, 
            noise_strength=noise_strength, scale_strength=scale_strength
        )
        
        print(f"🕐 TimeAwareMotionTrainer initialized:")
        print(f"   Temperature (augmentation): {temperature}")
        print(f"   Temporal weight: {temporal_weight}" + (" (DISABLED)" if temporal_weight == 0 else ""))
        print(f"   Use augmentation: {use_augmentation}")
        print(f"   Within-window consistency: {use_within_window_consistency}" + 
              (f" (weight={within_window_weight})" if use_within_window_consistency else ""))
        print(f"   Adjacent subwindow: {use_adjacent_subwindow}" +
              (f" (weight={adjacent_subwindow_weight}, τ={adjacent_temperature}, size={subwindow_size})" if use_adjacent_subwindow else ""))
        if augmentation_type == 'combined' and (noise_strength is not None or scale_strength is not None):
            print(f"   Augmentation type: {augmentation_type} (noise={noise_strength}, scale={scale_strength})")
        else:
            print(f"   Augmentation type: {augmentation_type} (strength={augmentation_strength})")
        print(f"   Mask same-track negatives: {mask_same_track}")
        print(f"   Augment temporal pairs: {augment_temporal_pairs}")
        print(f"   TensorBoard: {use_tensorboard}")
        print(f"   Save best model: {save_best_model}")
        if checkpoint_interval > 0:
            print(f"   Checkpoint interval: every {checkpoint_interval} epochs → {self.checkpoint_dir}")
        else:
            print(f"   Checkpoint interval: disabled (only final model saved)")
    
    def _train_epoch(self, dataloader):
        """Train one epoch with configurable contrastive losses."""
        self.model.train()
        epoch_loss = 0
        epoch_temporal = 0
        epoch_augmentation = 0
        epoch_within_window = 0
        epoch_adjacent_subwindow = 0
        
        for batch in dataloader:
            # Get window pairs and track IDs
            x_t = batch['features_t'].to(self.device)
            x_t1 = batch['features_t1'].to(self.device)
            track_ids = batch['unique_id']  # List of track IDs for masking
            
            # Optional: augment temporal pairs too
            if self.augment_temporal_pairs:
                x_t = self.augmentation_fn(x_t)
                x_t1 = self.augmentation_fn(x_t1)
            
            # Forward pass for full windows
            z_t = self.model(x_t)
            
            # Initialize loss components
            loss = torch.tensor(0.0, device=self.device)
            temporal_loss_val = 0.0
            aug_loss_val = 0.0
            within_window_loss_val = 0.0
            adjacent_subwindow_loss_val = 0.0
            
            # === TEMPORAL LOSS (legacy, often too easy with overlapping windows) ===
            if self.temporal_weight > 0:
                z_t1 = self.model(x_t1)
                temporal_loss = time_aware_contrastive_loss(
                    z_t, z_t1, self.temperature,
                    track_ids=track_ids,
                    mask_same_track=self.mask_same_track,
                )
                loss = loss + self.temporal_weight * temporal_loss
                temporal_loss_val = temporal_loss.item()
            
            # === AUGMENTATION LOSS (foundation, always recommended) ===
            if self.use_augmentation:
                x_t_orig = batch['features_t'].to(self.device)
                x_t_aug = self.augmentation_fn(x_t_orig)
                z_t_aug = self.model(x_t_aug)
                
                aug_loss = time_aware_contrastive_loss(
                    z_t, z_t_aug, self.temperature,
                    track_ids=track_ids,
                    mask_same_track=self.mask_same_track,
                )
                # Weight for augmentation is (1 - temporal_weight) when temporal is used,
                # otherwise it's the full loss
                if self.temporal_weight > 0:
                    aug_weight = 1.0 - self.temporal_weight
                else:
                    aug_weight = 1.0
                loss = loss + aug_weight * aug_loss
                aug_loss_val = aug_loss.item()
            
            # === WITHIN-WINDOW CONSISTENCY LOSS (interleaved frame views) ===
            if self.use_within_window_consistency:
                # Split window into non-overlapping views: even and odd frames
                # x_t shape: [B, T, 3] where T=60 typically
                x_t_orig = batch['features_t'].to(self.device)
                even_frames = x_t_orig[:, 0::2, :]  # [B, T/2, 3] - frames 0, 2, 4, ...
                odd_frames = x_t_orig[:, 1::2, :]   # [B, T/2, 3] - frames 1, 3, 5, ...
                
                # Forward pass through model for both views
                z_even = self.model(even_frames)
                z_odd = self.model(odd_frames)
                
                # Within-window consistency loss
                within_loss = within_window_contrastive_loss(
                    z_even, z_odd, self.temperature,
                    track_ids=track_ids,
                    mask_same_track=self.mask_same_track,
                )
                loss = loss + self.within_window_weight * within_loss
                within_window_loss_val = within_loss.item()
            
            # === ADJACENT SUBWINDOW CONTRASTIVE LOSS (NEW - temporal locality) ===
            if self.use_adjacent_subwindow:
                # Extract non-overlapping subwindows from full windows
                # x_t shape: [B, T, 3] where T=60 typically
                x_t_orig = batch['features_t'].to(self.device)
                B, T, F = x_t_orig.shape
                
                # Extract subwindows: [B, num_subs, subwindow_size, F]
                subwindows = extract_subwindows(x_t_orig, self.subwindow_size)
                B, num_subs, sub_size, F = subwindows.shape
                
                # Forward pass for all subwindows: [B*num_subs, embed_dim]
                z_subs = self.model(subwindows.reshape(B * num_subs, sub_size, F))
                z_subs = z_subs.reshape(B, num_subs, -1)  # [B, num_subs, embed_dim]
                
                # Compute adjacent subwindow contrastive loss
                adj_loss = adjacent_subwindow_contrastive_loss(
                    z_subs,
                    track_ids=track_ids,
                    temperature=self.adjacent_temperature,
                    mask_same_track=self.mask_same_track,
                )
                loss = loss + self.adjacent_subwindow_weight * adj_loss
                adjacent_subwindow_loss_val = adj_loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_temporal += temporal_loss_val
            epoch_augmentation += aug_loss_val
            epoch_within_window += within_window_loss_val
            epoch_adjacent_subwindow += adjacent_subwindow_loss_val
        
        n_batches = len(dataloader)
        return (
            epoch_loss / n_batches,
            epoch_temporal / n_batches,
            epoch_augmentation / n_batches,
            epoch_within_window / n_batches,
            epoch_adjacent_subwindow / n_batches,
        )
    
    def _validate_epoch(self, dataloader):
        """
        Validate one epoch using the SAME loss components as training.
        
        Returns individual loss components AND total, matching training.
        
        Returns
        -------
        tuple
            (total_loss, temporal_loss, augmentation_loss, within_window_loss, adjacent_subwindow_loss)
            Components that are disabled return 0.0
        """
        if dataloader is None:
            return None, 0.0, 0.0, 0.0, 0.0
        
        self.model.eval()
        epoch_loss = 0
        epoch_temporal = 0
        epoch_augmentation = 0
        epoch_within_window = 0
        epoch_adjacent_subwindow = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x_t = batch['features_t'].to(self.device)
                x_t1 = batch['features_t1'].to(self.device)
                track_ids = batch['unique_id']
                
                # Forward pass for full windows
                z_t = self.model(x_t)
                
                # Initialize loss
                loss = torch.tensor(0.0, device=self.device)
                temporal_loss_val = 0.0
                aug_loss_val = 0.0
                within_window_loss_val = 0.0
                adjacent_subwindow_loss_val = 0.0
                
                # === TEMPORAL LOSS (if enabled in training) ===
                if self.temporal_weight > 0:
                    z_t1 = self.model(x_t1)
                    temporal_loss = time_aware_contrastive_loss(
                        z_t, z_t1, self.temperature,
                        track_ids=track_ids,
                        mask_same_track=self.mask_same_track,
                    )
                    loss = loss + self.temporal_weight * temporal_loss
                    temporal_loss_val = temporal_loss.item()
                
                # === AUGMENTATION LOSS (if enabled in training) ===
                if self.use_augmentation:
                    x_t_aug = self.augmentation_fn(x_t)
                    z_t_aug = self.model(x_t_aug)
                    
                    aug_loss = time_aware_contrastive_loss(
                        z_t, z_t_aug, self.temperature,
                        track_ids=track_ids,
                        mask_same_track=self.mask_same_track,
                    )
                    # Same weight calculation as training
                    if self.temporal_weight > 0:
                        aug_weight = 1.0 - self.temporal_weight
                    else:
                        aug_weight = 1.0
                    loss = loss + aug_weight * aug_loss
                    aug_loss_val = aug_loss.item()
                
                # === WITHIN-WINDOW CONSISTENCY (if enabled in training) ===
                if self.use_within_window_consistency:
                    even_frames = x_t[:, 0::2, :]
                    odd_frames = x_t[:, 1::2, :]
                    
                    z_even = self.model(even_frames)
                    z_odd = self.model(odd_frames)
                    
                    within_loss = within_window_contrastive_loss(
                        z_even, z_odd, self.temperature,
                        track_ids=track_ids,
                        mask_same_track=self.mask_same_track,
                    )
                    loss = loss + self.within_window_weight * within_loss
                    within_window_loss_val = within_loss.item()
                
                # === ADJACENT SUBWINDOW CONTRASTIVE LOSS (if enabled in training) ===
                if self.use_adjacent_subwindow:
                    B, T, F = x_t.shape
                    subwindows = extract_subwindows(x_t, self.subwindow_size)
                    B, num_subs, sub_size, F = subwindows.shape
                    
                    z_subs = self.model(subwindows.reshape(B * num_subs, sub_size, F))
                    z_subs = z_subs.reshape(B, num_subs, -1)
                    
                    adj_loss = adjacent_subwindow_contrastive_loss(
                        z_subs,
                        track_ids=track_ids,
                        temperature=self.adjacent_temperature,
                        mask_same_track=self.mask_same_track,
                    )
                    loss = loss + self.adjacent_subwindow_weight * adj_loss
                    adjacent_subwindow_loss_val = adj_loss.item()
                
                epoch_loss += loss.item()
                epoch_temporal += temporal_loss_val
                epoch_augmentation += aug_loss_val
                epoch_within_window += within_window_loss_val
                epoch_adjacent_subwindow += adjacent_subwindow_loss_val
        
        n_batches = len(dataloader)
        return (
            epoch_loss / n_batches,
            epoch_temporal / n_batches,
            epoch_augmentation / n_batches,
            epoch_within_window / n_batches,
            epoch_adjacent_subwindow / n_batches,
        )
    
    def train(self, epochs=10):
        """
        Train for specified number of epochs.
        
        Parameters
        ----------
        epochs : int, default=10
            Number of training epochs
            
        Notes
        -----
        If save_best_model=True, the model state with lowest validation loss
        is tracked. Call restore_best_model() after training to load it.
        
        If checkpoint_interval > 0, checkpoints are saved every N epochs
        to checkpoint_dir with names like 'checkpoint_epoch_10.pt'.
        """
        import copy
        
        # Print training configuration summary
        print(f"\n🚀 Starting contrastive training for {epochs} epochs...")
        print(f"   Loss components:")
        if self.use_augmentation:
            print(f"   • Augmentation loss: ENABLED (τ={self.temperature})")
        if self.temporal_weight > 0:
            print(f"   • Temporal loss: ENABLED (weight={self.temporal_weight})")
        else:
            print(f"   • Temporal loss: DISABLED")
        if self.use_within_window_consistency:
            print(f"   • Within-window consistency: ENABLED (weight={self.within_window_weight})")
        else:
            print(f"   • Within-window consistency: DISABLED")
        if self.use_adjacent_subwindow:
            print(f"   • Adjacent subwindow: ENABLED (weight={self.adjacent_subwindow_weight}, τ={self.adjacent_temperature}, size={self.subwindow_size})")
        else:
            print(f"   • Adjacent subwindow: DISABLED")
        if self.save_best_model:
            print(f"   • Best model tracking: ENABLED (based on validation loss)")
        if self.checkpoint_interval > 0:
            print(f"   • Checkpoints: every {self.checkpoint_interval} epochs → {self.checkpoint_dir}")
        if self.early_stopping_patience > 0:
            print(f"   • Early stopping: after {self.early_stopping_patience} epochs without improvement")
        print()
        
        for epoch in range(epochs):
            # Global epoch = offset + local epoch (for correct TensorBoard/checkpoint numbering)
            global_epoch = self.epoch_offset + epoch
            
            # Train
            train_loss, temporal_loss, aug_loss, within_loss, adj_loss = self._train_epoch(self.train_dataloader)
            self.train_losses.append(train_loss)
            self.temporal_losses.append(temporal_loss)
            self.augmentation_losses.append(aug_loss)
            self.within_window_losses.append(within_loss)
            self.adjacent_subwindow_losses.append(adj_loss)
            
            # Validate (returns same components as training)
            val_loss, val_temporal, val_aug, val_within, val_adj = self._validate_epoch(self.val_dataloader)
            if val_loss is not None:
                self.val_losses.append(val_loss)
                self.val_temporal_losses.append(val_temporal)
                self.val_augmentation_losses.append(val_aug)
                self.val_within_window_losses.append(val_within)
                self.val_adjacent_subwindow_losses.append(val_adj)
            
            # Track best model (based on validation loss) and early stopping
            best_marker = ""
            if self.save_best_model and val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = global_epoch  # Use global epoch for best_epoch tracking
                    self.epochs_without_improvement = 0
                    best_marker = " ⭐ NEW BEST"
                else:
                    self.epochs_without_improvement += 1
            
            # Save checkpoint at interval (use global epoch for filename)
            checkpoint_marker = ""
            if self.checkpoint_interval > 0 and (global_epoch + 1) % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{global_epoch+1}.pt")
                self.save_checkpoint(checkpoint_path, global_epoch)
                checkpoint_marker = f" 💾 Saved checkpoint"
            
            # Update scheduler
            if self.use_scheduler:
                self.scheduler.step(val_loss if val_loss else train_loss)
            
            # TensorBoard logging (use global_epoch for correct step numbers)
            if self.writer is not None:
                # Training losses
                self.writer.add_scalar('Loss/train_total', train_loss, global_epoch)
                if self.temporal_weight > 0:
                    self.writer.add_scalar('Loss/train_temporal', temporal_loss, global_epoch)
                if self.use_augmentation:
                    self.writer.add_scalar('Loss/train_augmentation', aug_loss, global_epoch)
                if self.use_within_window_consistency:
                    self.writer.add_scalar('Loss/train_within_window', within_loss, global_epoch)
                if self.use_adjacent_subwindow:
                    self.writer.add_scalar('Loss/train_adjacent_subwindow', adj_loss, global_epoch)
                
                # Validation losses (same components as training)
                if val_loss is not None:
                    self.writer.add_scalar('Loss/val_total', val_loss, global_epoch)
                    if self.temporal_weight > 0:
                        self.writer.add_scalar('Loss/val_temporal', val_temporal, global_epoch)
                    if self.use_augmentation:
                        self.writer.add_scalar('Loss/val_augmentation', val_aug, global_epoch)
                    if self.use_within_window_consistency:
                        self.writer.add_scalar('Loss/val_within_window', val_within, global_epoch)
                    if self.use_adjacent_subwindow:
                        self.writer.add_scalar('Loss/val_adjacent_subwindow', val_adj, global_epoch)
                
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], global_epoch)
            
            # Print progress with active loss components
            lr = self.optimizer.param_groups[0]['lr']
            
            # Training components
            train_parts = []
            if self.temporal_weight > 0:
                train_parts.append(f"T:{temporal_loss:.4f}")
            if self.use_augmentation:
                train_parts.append(f"A:{aug_loss:.4f}")
            if self.use_within_window_consistency:
                train_parts.append(f"W:{within_loss:.4f}")
            if self.use_adjacent_subwindow:
                train_parts.append(f"S:{adj_loss:.4f}")
            train_str = ", ".join(train_parts) if train_parts else "no components"
            
            # Validation components (matching training)
            if val_loss is not None:
                val_parts = []
                if self.temporal_weight > 0:
                    val_parts.append(f"T:{val_temporal:.4f}")
                if self.use_augmentation:
                    val_parts.append(f"A:{val_aug:.4f}")
                if self.use_within_window_consistency:
                    val_parts.append(f"W:{val_within:.4f}")
                if self.use_adjacent_subwindow:
                    val_parts.append(f"S:{val_adj:.4f}")
                val_str = ", ".join(val_parts) if val_parts else ""
                
                print(f"Epoch {global_epoch+1:3d}: Train={train_loss:.4f} ({train_str}), Val={val_loss:.4f} ({val_str}), LR={lr:.2e}{best_marker}{checkpoint_marker}")
            else:
                print(f"Epoch {global_epoch+1:3d}: Train={train_loss:.4f} ({train_str}), LR={lr:.2e}{checkpoint_marker}")
            
            # Early stopping check
            if self.early_stopping_patience > 0 and self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n⏹️  Early stopping: No improvement for {self.early_stopping_patience} epochs")
                print(f"   Stopped at epoch {global_epoch + 1}")
                break
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        # Report best model info
        print("\n✅ Training completed!")
        if self.save_best_model and self.best_model_state is not None:
            print(f"⭐ Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch + 1}")
            print(f"   💡 Call trainer.restore_best_model() to load the optimal weights")
    
    def extract_embeddings(self, dataloader):
        """
        Extract embeddings from a dataloader.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader with trajectory data
            
        Returns
        -------
        np.ndarray
            Extracted embeddings
        """
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                if 'features' in batch:
                    x = batch['features'].to(self.device)
                else:
                    x = batch['features_t'].to(self.device)
                
                z = self.model(x)
                all_embeddings.append(z.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def restore_best_model(self):
        """
        Restore model weights to the state with lowest validation loss.
        
        This should be called after training if you want to use the model
        from the optimal epoch rather than the final epoch.
        
        Returns
        -------
        bool
            True if restoration was successful, False if no best model was saved
            
        Examples
        --------
        >>> trainer.train(epochs=100)
        >>> # Training complete, model is at epoch 100
        >>> trainer.restore_best_model()
        >>> # Now model is at epoch with lowest val loss (e.g., epoch 15)
        >>> embeddings = trainer.extract_embeddings(test_loader)
        """
        if self.best_model_state is None:
            print("⚠️ No best model state saved. Make sure:")
            print("   1. save_best_model=True was set")
            print("   2. A validation dataloader was provided")
            print("   3. Training was run")
            return False
        
        self.model.load_state_dict(self.best_model_state)
        print(f"✅ Restored model to epoch {self.best_epoch + 1}")
        print(f"   Validation loss at that epoch: {self.best_val_loss:.4f}")
        return True
    
    def save_checkpoint(self, path, epoch):
        """
        Save a checkpoint at a specific epoch.
        
        Checkpoints include full training state, allowing training to be resumed.
        
        Parameters
        ----------
        path : str
            Path to save the checkpoint file
        epoch : int
            Current epoch number (0-indexed)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Training losses
            'train_losses': self.train_losses,
            'temporal_losses': self.temporal_losses,
            'augmentation_losses': self.augmentation_losses,
            'within_window_losses': self.within_window_losses,
            'adjacent_subwindow_losses': self.adjacent_subwindow_losses,
            # Validation losses (component-wise)
            'val_losses': self.val_losses,
            'val_temporal_losses': self.val_temporal_losses,
            'val_augmentation_losses': self.val_augmentation_losses,
            'val_within_window_losses': self.val_within_window_losses,
            'val_adjacent_subwindow_losses': self.val_adjacent_subwindow_losses,
            # Best model tracking
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            # Early stopping
            'epochs_without_improvement': self.epochs_without_improvement,
            # Config
            'temperature': self.temperature,
            'temporal_weight': self.temporal_weight,
            'use_within_window_consistency': self.use_within_window_consistency,
            'within_window_weight': self.within_window_weight,
        }
        
        # Save scheduler state if available
        if self.use_scheduler and self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Also save best model state if available
        if self.best_model_state is not None:
            checkpoint['best_model_state'] = self.best_model_state
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path, restore_best=False):
        """
        Load a checkpoint and optionally restore to best model state.
        
        Parameters
        ----------
        path : str
            Path to the checkpoint file
        restore_best : bool, default=False
            If True and checkpoint contains best_model_state, restore to that.
            If False, restore to the state at the checkpoint epoch.
            
        Returns
        -------
        int
            The epoch number of the loaded checkpoint
            
        Examples
        --------
        >>> # Load checkpoint and continue from that epoch
        >>> epoch = trainer.load_checkpoint("checkpoint_epoch_50.pt")
        >>> trainer.train(epochs=50)  # Continue for 50 more epochs
        >>> 
        >>> # Load checkpoint and get best model from it
        >>> trainer.load_checkpoint("checkpoint_epoch_100.pt", restore_best=True)
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore training state
        self.train_losses = checkpoint.get('train_losses', [])
        self.temporal_losses = checkpoint.get('temporal_losses', [])
        self.augmentation_losses = checkpoint.get('augmentation_losses', [])
        self.within_window_losses = checkpoint.get('within_window_losses', [])
        
        # Restore validation state (component-wise)
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_temporal_losses = checkpoint.get('val_temporal_losses', [])
        self.val_augmentation_losses = checkpoint.get('val_augmentation_losses', [])
        self.val_within_window_losses = checkpoint.get('val_within_window_losses', [])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', -1)
        
        # Restore best model state if available
        if 'best_model_state' in checkpoint:
            self.best_model_state = checkpoint['best_model_state']
        
        # Restore optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore model (either checkpoint state or best state)
        if restore_best and 'best_model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['best_model_state'])
            print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
            print(f"   Restored BEST model from epoch {self.best_epoch + 1} (val_loss={self.best_val_loss:.4f})")
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
            if self.best_epoch >= 0:
                print(f"   Best validation loss was {self.best_val_loss:.4f} at epoch {self.best_epoch + 1}")
                print(f"   💡 Call trainer.restore_best_model() or load with restore_best=True")
        
        return checkpoint['epoch']
    
    def save_model(self, path):
        """
        Save model and training state.
        
        Includes best model state if save_best_model was enabled during training.
        
        Parameters
        ----------
        path : str
            Path to save the checkpoint
        """
        # Extract architecture from model (if TransformerMotionEncoder)
        architecture = {}
        if hasattr(self.model, 'input_proj'):
            architecture['embed_dim'] = self.model.input_proj.out_features
            architecture['input_dim'] = self.model.input_proj.in_features
        if hasattr(self.model, 'transformer_encoder'):
            encoder = self.model.transformer_encoder
            architecture['num_layers'] = encoder.num_layers
            if hasattr(encoder, 'layers') and len(encoder.layers) > 0:
                layer = encoder.layers[0]
                if hasattr(layer, 'self_attn'):
                    architecture['num_heads'] = layer.self_attn.num_heads
                if hasattr(layer, 'linear1'):
                    architecture['ff_dim'] = layer.linear1.out_features
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Training losses
            'train_losses': self.train_losses,
            'temporal_losses': self.temporal_losses,
            'augmentation_losses': self.augmentation_losses,
            'within_window_losses': self.within_window_losses,
            # Validation losses (component-wise)
            'val_losses': self.val_losses,
            'val_temporal_losses': self.val_temporal_losses,
            'val_augmentation_losses': self.val_augmentation_losses,
            'val_within_window_losses': self.val_within_window_losses,
            # Config
            'temperature': self.temperature,
            'temporal_weight': self.temporal_weight,
            'use_within_window_consistency': self.use_within_window_consistency,
            'within_window_weight': self.within_window_weight,
            'training_type': 'contrastive_learning',
            'architecture': architecture,
            # Best model tracking info
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }
        
        # Include best model state if available
        if self.best_model_state is not None:
            save_dict['best_model_state'] = self.best_model_state
        
        torch.save(save_dict, path)
        print(f"✅ Model saved to: {path}")
        if architecture:
            print(f"   Architecture: embed_dim={architecture.get('embed_dim')}, "
                  f"heads={architecture.get('num_heads')}, "
                  f"ff_dim={architecture.get('ff_dim')}, "
                  f"layers={architecture.get('num_layers')}")
        print(f"   Training config: temporal_weight={self.temporal_weight}, "
              f"within_window={self.use_within_window_consistency}")
        if self.best_model_state is not None:
            print(f"   ⭐ Best model included: epoch {self.best_epoch + 1}, val_loss={self.best_val_loss:.4f}")
            print(f"   💡 After loading, call trainer.restore_best_model() to use optimal weights")
    
    def load_model(self, path, restore_best=False):
        """
        Load model from checkpoint.
        
        Parameters
        ----------
        path : str
            Path to the checkpoint file
        restore_best : bool, default=False
            If True and checkpoint contains best_model_state, restore to that.
            If False, restore to the final model state.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Check architecture if saved
        if 'architecture' in checkpoint:
            saved_arch = checkpoint['architecture']
            print(f"   Saved architecture: embed_dim={saved_arch.get('embed_dim')}, "
                  f"heads={saved_arch.get('num_heads')}, "
                  f"ff_dim={saved_arch.get('ff_dim')}, "
                  f"layers={saved_arch.get('num_layers')}")
        
        # Load training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.temporal_losses = checkpoint.get('temporal_losses', [])
        self.augmentation_losses = checkpoint.get('augmentation_losses', [])
        self.within_window_losses = checkpoint.get('within_window_losses', [])
        
        # Load validation history (component-wise)
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_temporal_losses = checkpoint.get('val_temporal_losses', [])
        self.val_augmentation_losses = checkpoint.get('val_augmentation_losses', [])
        self.val_within_window_losses = checkpoint.get('val_within_window_losses', [])
        
        # Load best model tracking info
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', -1)
        if 'best_model_state' in checkpoint:
            self.best_model_state = checkpoint['best_model_state']
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load model (either final state or best state)
        if restore_best and 'best_model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['best_model_state'])
            print(f"✅ Model loaded from: {path}")
            print(f"   Restored BEST model from epoch {self.best_epoch + 1} (val_loss={self.best_val_loss:.4f})")
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from: {path}")
        print(f"   Previous epochs: {len(self.train_losses)}")
        if self.best_epoch >= 0:
            print(f"   ⭐ Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch + 1}")
            print(f"   💡 Call trainer.restore_best_model() or reload with restore_best=True")
        
        if 'use_within_window_consistency' in checkpoint:
            print(f"   Within-window consistency: {checkpoint.get('use_within_window_consistency')}")
    
    def plot_training_curves(self):
        """Plot training curves showing loss progression for both train and validation."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # === TOP LEFT: Total Loss (Train vs Val) ===
        ax1 = axes[0, 0]
        ax1.plot(self.train_losses, label='Train Total', color='blue', linewidth=2)
        if self.val_losses:
            ax1.plot(self.val_losses, label='Val Total', color='red', linewidth=2)
        if self.best_epoch >= 0:
            ax1.axvline(self.best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best (epoch {self.best_epoch+1})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Total Loss: Train vs Validation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # === TOP RIGHT: Training Components ===
        ax2 = axes[0, 1]
        if self.temporal_losses and any(l > 0 for l in self.temporal_losses):
            ax2.plot(self.temporal_losses, label='Train Temporal', color='green', linewidth=2)
        if self.augmentation_losses and any(l > 0 for l in self.augmentation_losses):
            ax2.plot(self.augmentation_losses, label='Train Augmentation', color='orange', linewidth=2)
        if self.within_window_losses and any(l > 0 for l in self.within_window_losses):
            ax2.plot(self.within_window_losses, label='Train Within-Window', color='purple', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # === BOTTOM LEFT: Validation Components ===
        ax3 = axes[1, 0]
        if self.val_temporal_losses and any(l > 0 for l in self.val_temporal_losses):
            ax3.plot(self.val_temporal_losses, label='Val Temporal', color='green', linewidth=2, linestyle='--')
        if self.val_augmentation_losses and any(l > 0 for l in self.val_augmentation_losses):
            ax3.plot(self.val_augmentation_losses, label='Val Augmentation', color='orange', linewidth=2, linestyle='--')
        if self.val_within_window_losses and any(l > 0 for l in self.val_within_window_losses):
            ax3.plot(self.val_within_window_losses, label='Val Within-Window', color='purple', linewidth=2, linestyle='--')
        if self.best_epoch >= 0:
            ax3.axvline(self.best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best (epoch {self.best_epoch+1})')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Validation Loss Components')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # === BOTTOM RIGHT: Train vs Val per Component ===
        ax4 = axes[1, 1]
        # Augmentation comparison
        if self.augmentation_losses and any(l > 0 for l in self.augmentation_losses):
            ax4.plot(self.augmentation_losses, label='Train Aug', color='orange', linewidth=2)
        if self.val_augmentation_losses and any(l > 0 for l in self.val_augmentation_losses):
            ax4.plot(self.val_augmentation_losses, label='Val Aug', color='orange', linewidth=2, linestyle='--')
        # Within-window comparison
        if self.within_window_losses and any(l > 0 for l in self.within_window_losses):
            ax4.plot(self.within_window_losses, label='Train Within', color='purple', linewidth=2)
        if self.val_within_window_losses and any(l > 0 for l in self.val_within_window_losses):
            ax4.plot(self.val_within_window_losses, label='Val Within', color='purple', linewidth=2, linestyle='--')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Component Comparison: Train (solid) vs Val (dashed)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# TEMPORAL SMOOTHNESS EVALUATION FUNCTIONS
# =============================================================================

def evaluate_temporal_smoothness(model, dataloader, device="cuda"):
    """
    Evaluate temporal smoothness of embeddings from a trained model.
    
    Computes the Dynamic Range (DR) metric from DynaCLR, which measures
    how well the model preserves temporal continuity in embeddings.
    Adjacent windows should have higher similarity than random pairs.
    
    Parameters
    ----------
    model : nn.Module
        Trained encoder model (e.g., TransformerMotionEncoder)
    dataloader : DataLoader
        DataLoader yielding batches with 'features_t' and 'features_t1' keys
        (from TimeAwareTrajectoryDataset)
    device : str, default="cuda"
        Device to run evaluation on
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'adjacent_similarities': np.ndarray of similarities between adjacent windows
        - 'random_similarities': np.ndarray of similarities between random pairs
        - 'dynamic_range': DR = median(adjacent) - median(random)
        - 'adjacent_mean': Mean adjacent similarity
        - 'adjacent_std': Std of adjacent similarity
        - 'random_mean': Mean random similarity
        - 'random_std': Std of random similarity
        
    Examples
    --------
    >>> results = evaluate_temporal_smoothness(trainer.model, test_loader)
    >>> print(f"Dynamic Range: {results['dynamic_range']:.3f}")
    
    Notes
    -----
    Interpretation of Dynamic Range (DR):
    - DR > 0.4: Excellent temporal smoothness
    - DR > 0.2: Good temporal smoothness
    - DR < 0.2: Poor temporal regularization
    """
    import torch.nn.functional as F
    
    print("📊 Evaluating temporal smoothness of embeddings...")
    
    model.eval()
    model.to(device)
    
    adjacent_similarities = []
    random_similarities = []
    
    with torch.no_grad():
        for batch in dataloader:
            x_t = batch['features_t'].to(device)
            x_t1 = batch['features_t1'].to(device)
            
            z_t = model(x_t)
            z_t1 = model(x_t1)
            
            # Normalize embeddings
            z_t = F.normalize(z_t, dim=1)
            z_t1 = F.normalize(z_t1, dim=1)
            
            # Adjacent similarities (temporal positives)
            adj_sim = F.cosine_similarity(z_t, z_t1, dim=1)
            adjacent_similarities.extend(adj_sim.cpu().numpy())
            
            # Random similarities (shuffle z_t1)
            idx = torch.randperm(z_t1.size(0))
            z_random = z_t1[idx]
            rand_sim = F.cosine_similarity(z_t, z_random, dim=1)
            random_similarities.extend(rand_sim.cpu().numpy())
    
    adjacent_similarities = np.array(adjacent_similarities)
    random_similarities = np.array(random_similarities)
    
    # Calculate Dynamic Range (DR) - key metric from DynaCLR
    dr = np.median(adjacent_similarities) - np.median(random_similarities)
    
    # Print results
    print(f"\n🎯 Temporal Smoothness Metrics:")
    print(f"   Adjacent similarity: {np.mean(adjacent_similarities):.3f} ± {np.std(adjacent_similarities):.3f}")
    print(f"   Random similarity:   {np.mean(random_similarities):.3f} ± {np.std(random_similarities):.3f}")
    print(f"   Dynamic Range (DR):  {dr:.3f}")
    print(f"")
    
    # Interpretation
    if dr >= 0.4:
        quality = "✅ Excellent"
    elif dr >= 0.2:
        quality = "✅ Good"
    else:
        quality = "⚠️ Poor"
    print(f"   Temporal smoothness: {quality}")
    print(f"   (DR > 0.2: Good, DR > 0.4: Excellent)")
    
    return {
        'adjacent_similarities': adjacent_similarities,
        'random_similarities': random_similarities,
        'dynamic_range': dr,
        'adjacent_mean': np.mean(adjacent_similarities),
        'adjacent_std': np.std(adjacent_similarities),
        'random_mean': np.mean(random_similarities),
        'random_std': np.std(random_similarities),
    }


def plot_temporal_smoothness(results, save_path=None):
    """
    Plot temporal smoothness evaluation results.
    
    Creates a two-panel figure showing:
    1. Histogram of adjacent vs random similarities
    2. Box plot comparison
    
    Parameters
    ----------
    results : dict
        Results dictionary from evaluate_temporal_smoothness()
    save_path : str or Path, optional
        Path to save the figure (as SVG with transparent background)
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
        
    Examples
    --------
    >>> results = evaluate_temporal_smoothness(model, test_loader)
    >>> fig = plot_temporal_smoothness(results, save_path="./temporal_smoothness.svg")
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    adjacent_similarities = results['adjacent_similarities']
    random_similarities = results['random_similarities']
    dr = results['dynamic_range']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(adjacent_similarities, bins=50, alpha=0.7, 
             label=f'Adjacent (μ={np.mean(adjacent_similarities):.3f})', color='green')
    ax1.hist(random_similarities, bins=50, alpha=0.7, 
             label=f'Random (μ={np.mean(random_similarities):.3f})', color='red')
    ax1.axvline(np.median(adjacent_similarities), color='green', linestyle='--', linewidth=2)
    ax1.axvline(np.median(random_similarities), color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Cosine Similarity', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'Similarity Distribution (DR = {dr:.3f})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot([adjacent_similarities, random_similarities], 
                      labels=['Adjacent\n(Temporal +)', 'Random\n(Temporal -)'],
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Similarity Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path specified
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set transparent background with visible axes
        fig.patch.set_facecolor('none')
        for ax in axes:
            ax.set_facecolor('none')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.8)
            ax.tick_params(colors='black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            ax.title.set_color('black')
        
        fig.savefig(save_path, format='svg', transparent=True, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    
    return fig


def analyze_frame_importance(
    model,
    dataloader,
    device="cuda",
    method="gradient",
    n_samples=1000,
    save_path=None
):
    """
    Analyze which frames the model considers most important for embeddings.
    
    Uses gradient-based or leave-one-out importance to understand what
    temporal patterns the model has learned to focus on.
    
    Parameters
    ----------
    model : nn.Module
        Trained encoder model (e.g., TransformerMotionEncoder)
    dataloader : DataLoader
        DataLoader with trajectory windows
    device : str, default="cuda"
        Device to run analysis on
    method : str, default="gradient"
        Importance method: 'gradient' (fast) or 'leave_one_out' (slow but intuitive)
    n_samples : int, default=1000
        Number of samples to analyze (for speed)
    save_path : str or Path, optional
        Path to save the figure
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'importance_scores': np.ndarray of shape (n_samples, window_size)
        - 'mean_importance': Average importance per frame position
        - 'std_importance': Std of importance per frame position
        - 'method': Method used
        
    Examples
    --------
    >>> results = analyze_frame_importance(trainer.model, test_loader, n_samples=500)
    >>> print(f"Most important frame: {np.argmax(results['mean_importance'])}")
    
    Notes
    -----
    Interpretation:
    - High importance at start: model focuses on initial motion
    - High importance at end: model focuses on final motion state
    - Uniform importance: model uses entire trajectory
    - Peaks in middle: specific events matter most
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    print(f"🔍 Analyzing frame importance using '{method}' method...")
    print(f"   Analyzing {n_samples} samples...")
    
    model.eval()
    model.to(device)
    
    all_importance = []
    n_processed = 0
    
    for batch in dataloader:
        if n_processed >= n_samples:
            break
            
        # Get features
        if 'features' in batch:
            x = batch['features'].to(device)
        else:
            x = batch['features_t'].to(device)
        
        batch_size = x.shape[0]
        remaining = n_samples - n_processed
        if batch_size > remaining:
            x = x[:remaining]
        
        # Extract importance using model's built-in method
        if hasattr(model, 'extract_attention_weights'):
            importance = model.extract_attention_weights(x, method=method)
            all_importance.append(importance.cpu().numpy())
        else:
            print("⚠️ Model doesn't have extract_attention_weights method")
            return None
        
        n_processed += x.shape[0]
    
    # Combine results
    importance_scores = np.vstack(all_importance)
    mean_importance = np.mean(importance_scores, axis=0)
    std_importance = np.std(importance_scores, axis=0)
    
    # Normalize for visualization
    mean_importance_norm = (mean_importance - mean_importance.min()) / (mean_importance.max() - mean_importance.min() + 1e-8)
    
    print(f"\n🎯 Frame Importance Analysis:")
    print(f"   Window size: {len(mean_importance)} frames")
    print(f"   Most important frame: {np.argmax(mean_importance)} (position)")
    print(f"   Least important frame: {np.argmin(mean_importance)} (position)")
    
    # Check for temporal patterns
    first_third = np.mean(mean_importance[:len(mean_importance)//3])
    middle_third = np.mean(mean_importance[len(mean_importance)//3:2*len(mean_importance)//3])
    last_third = np.mean(mean_importance[2*len(mean_importance)//3:])
    
    print(f"\n   Temporal focus:")
    print(f"      Start (first 1/3):  {first_third:.4f}")
    print(f"      Middle (middle 1/3): {middle_third:.4f}")
    print(f"      End (last 1/3):     {last_third:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean importance across frames
    ax1 = axes[0]
    frames = np.arange(len(mean_importance))
    ax1.fill_between(frames, mean_importance - std_importance, mean_importance + std_importance, 
                     alpha=0.3, color='blue')
    ax1.plot(frames, mean_importance, 'b-', linewidth=2, label='Mean importance')
    ax1.axhline(np.mean(mean_importance), color='red', linestyle='--', alpha=0.7, label='Overall mean')
    ax1.set_xlabel('Frame Position', fontsize=12)
    ax1.set_ylabel('Importance Score', fontsize=12)
    ax1.set_title(f'Frame Importance ({method} method)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Heatmap of individual samples
    ax2 = axes[1]
    # Show subset of samples as heatmap
    n_show = min(50, len(importance_scores))
    im = ax2.imshow(importance_scores[:n_show], aspect='auto', cmap='viridis')
    ax2.set_xlabel('Frame Position', fontsize=12)
    ax2.set_ylabel('Sample', fontsize=12)
    ax2.set_title(f'Individual Sample Importance (n={n_show})', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Importance')
    
    plt.tight_layout()
    
    # Save if path specified
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.patch.set_facecolor('none')
        for ax in axes:
            ax.set_facecolor('none')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.8)
            ax.tick_params(colors='black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            ax.title.set_color('black')
        
        fig.savefig(save_path, format='svg', transparent=True, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    
    return {
        'importance_scores': importance_scores,
        'mean_importance': mean_importance,
        'std_importance': std_importance,
        'method': method,
    }


def visualize_track_attention(
    model,
    dataset,
    device="cuda",
    n_examples=6,
    method="leave_one_out",
    figsize=(14, 10),
    save_path=None
):
    """
    Visualize what motion patterns trigger model attention on example tracks.
    
    Shows trajectory segments color-coded by importance (no dots, just lines).
    All plots use the same axis range for easy comparison.
    
    Parameters
    ----------
    model : nn.Module
        Trained encoder model
    dataset : TimeAwareTrajectoryDataset or similar
        Dataset with trajectory windows (needs 'features_t' or 'features')
    device : str, default="cuda"
        Device for computation
    n_examples : int, default=6
        Number of example tracks to visualize
    method : str, default="leave_one_out"
        Importance method: 'leave_one_out' (intuitive) or 'gradient' (fast)
    figsize : tuple, default=(14, 10)
        Figure size
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    dict
        Results containing:
        - 'examples': List of dicts with track data and importance
        - 'feature_correlations': Correlation between importance and motion features
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from pathlib import Path
    import torch.nn.functional as F
    
    print(f"🔍 Visualizing track attention using '{method}' method...")
    print(f"   Analyzing {n_examples} example tracks...")
    
    model.eval()
    model.to(device)
    
    # Collect examples
    examples = []
    all_importance = []
    all_speeds = []
    all_dir_changes = []
    
    # Get random indices
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(n_examples, len(dataset)), replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        
        # Get features
        if 'features' in sample:
            features = sample['features'].unsqueeze(0).to(device)
        else:
            features = sample['features_t'].unsqueeze(0).to(device)
        
        # Compute importance using model method
        if hasattr(model, 'extract_attention_weights'):
            importance = model.extract_attention_weights(features, method=method)
            importance = importance.squeeze().cpu().numpy()
        else:
            print("⚠️ Model doesn't have extract_attention_weights method")
            return None
        
        # Extract motion features from the window
        feat_np = features.squeeze().cpu().numpy()  # [T, 3] - dx, dy, direction
        dx = feat_np[:, 0]
        dy = feat_np[:, 1]
        
        # Reconstruct trajectory from displacements (centered at origin)
        x = np.concatenate([[0], np.cumsum(dx)])
        y = np.concatenate([[0], np.cumsum(dy)])
        
        # Compute speed and direction change
        speed = np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx)
        dir_change = np.abs(np.diff(direction, prepend=direction[0]))
        dir_change = np.minimum(dir_change, 2*np.pi - dir_change)  # Handle wraparound
        
        # Compute acceleration
        accel = np.abs(np.diff(speed, prepend=speed[0]))
        
        # Store
        examples.append({
            'x': x,
            'y': y,
            'dx': dx,
            'dy': dy,
            'speed': speed,
            'dir_change': dir_change,
            'acceleration': accel,
            'importance': importance,
            'unique_id': sample.get('unique_id', f'track_{idx}'),
            'condition': sample.get('condition', 'unknown'),
        })
        
        all_importance.extend(importance)
        all_speeds.extend(speed)
        all_dir_changes.extend(dir_change)
    
    # Compute correlations
    all_importance = np.array(all_importance)
    all_speeds = np.array(all_speeds)
    all_dir_changes = np.array(all_dir_changes)
    
    speed_corr = np.corrcoef(all_importance, all_speeds)[0, 1]
    dir_corr = np.corrcoef(all_importance, all_dir_changes)[0, 1]
    
    print(f"\n🎯 Motion Feature Correlations with Importance:")
    print(f"   Speed ↔ Importance:     r = {speed_corr:+.3f}")
    print(f"   Dir Change ↔ Importance: r = {dir_corr:+.3f}")
    
    # Find global axis range across all tracks (for uniform plotting)
    all_x = np.concatenate([ex['x'] for ex in examples])
    all_y = np.concatenate([ex['y'] for ex in examples])
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    max_range = max(x_range, y_range) * 1.1  # 10% padding
    x_center = (all_x.max() + all_x.min()) / 2
    y_center = (all_y.max() + all_y.min()) / 2
    
    # For individual plots, use per-track centering but same scale
    per_track_range = max([
        max(ex['x'].max() - ex['x'].min(), ex['y'].max() - ex['y'].min())
        for ex in examples
    ]) * 1.2  # 20% padding
    
    # Create visualization
    n_cols = 3
    n_rows = (n_examples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_examples > 1 else [axes]
    
    for i, (ax, ex) in enumerate(zip(axes, examples)):
        x, y = ex['x'], ex['y']
        importance = ex['importance']
        
        # Normalize importance for coloring (global across this track)
        imp_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        # Create line segments colored by importance
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Use segment importance (average of two endpoints) - need to match segment count
        seg_importance = (imp_norm[:-1] + imp_norm[1:]) / 2 if len(imp_norm) == len(x) - 1 else imp_norm
        if len(seg_importance) > len(segments):
            seg_importance = seg_importance[:len(segments)]
        
        lc = LineCollection(segments, cmap='hot', linewidths=2.5)
        lc.set_array(seg_importance)
        ax.add_collection(lc)
        
        # Mark start and end only
        ax.scatter(x[0], y[0], c='limegreen', s=80, marker='^', edgecolors='black', 
                   linewidths=1, zorder=6, label='Start')
        ax.scatter(x[-1], y[-1], c='dodgerblue', s=80, marker='s', edgecolors='black', 
                   linewidths=1, zorder=6, label='End')
        
        # Set equal square axes centered on this track
        x_center_track = (x.max() + x.min()) / 2
        y_center_track = (y.max() + y.min()) / 2
        half_range = per_track_range / 2
        
        ax.set_xlim(x_center_track - half_range, x_center_track + half_range)
        ax.set_ylim(y_center_track - half_range, y_center_track + half_range)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x (µm)', fontsize=9)
        ax.set_ylabel('y (µm)', fontsize=9)
        ax.set_title(f"{ex['condition'][:30]}\n{ex['unique_id'][:25]}", fontsize=9)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(loc='upper right', fontsize=7)
    
    # Hide empty subplots
    for ax in axes[len(examples):]:
        ax.set_visible(False)
    
    # Add colorbar on the right side, outside the plots
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Normalized Importance')
    
    fig.suptitle(f'Track Attention ({method})\n'
                 f'Speed↔Imp: r={speed_corr:.2f}, DirChange↔Imp: r={dir_corr:.2f}',
                 fontsize=12, fontweight='bold', y=0.98)
    
    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format='svg', transparent=True, bbox_inches='tight', dpi=150)
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    
    return {
        'examples': examples,
        'feature_correlations': {
            'speed': speed_corr,
            'direction_change': dir_corr,
        },
        'method': method,
    }


def analyze_attention_by_motion_state(
    model,
    dataset,
    device="cuda",
    n_samples=500,
    method="gradient",
    save_path=None,
):
    """
    Analyze how attention correlates with motion features using scatter plots.
    
    Creates scatter plots of attention vs speed, direction change, and acceleration.
    Includes trend lines and correlation statistics.
    
    Parameters
    ----------
    model : nn.Module
        Trained encoder model
    dataset : Dataset
        Trajectory dataset
    device : str, default="cuda"
        Computation device
    n_samples : int, default=500
        Number of windows to analyze
    method : str, default="gradient"
        Importance method
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    dict
        Analysis results with correlations and raw data for further analysis
        
    Examples
    --------
    >>> results = analyze_attention_by_motion_state(model, dataset)
    >>> print(f"Speed correlation: {results['speed_correlation']:.3f}")
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    from pathlib import Path
    
    print(f"📊 Analyzing attention vs motion features ({n_samples} samples)...")
    
    model.eval()
    model.to(device)
    
    all_speeds = []
    all_dir_changes = []
    all_accelerations = []
    all_step_lengths = []
    all_importance = []
    
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    for idx in tqdm(indices, desc="Extracting features"):
        sample = dataset[idx]
        
        if 'features' in sample:
            features = sample['features'].unsqueeze(0).to(device)
        else:
            features = sample['features_t'].unsqueeze(0).to(device)
        
        # Get importance
        importance = model.extract_attention_weights(features, method=method)
        importance = importance.squeeze().cpu().numpy()
        
        # Extract motion features
        feat_np = features.squeeze().cpu().numpy()
        dx, dy = feat_np[:, 0], feat_np[:, 1]
        
        # Basic features
        step_length = np.sqrt(dx**2 + dy**2)
        speed = step_length  # Same as step length if dt=1
        direction = np.arctan2(dy, dx)
        dir_change = np.abs(np.diff(direction, prepend=direction[0]))
        dir_change = np.minimum(dir_change, 2*np.pi - dir_change)
        acceleration = np.abs(np.diff(speed, prepend=speed[0]))
        
        all_speeds.extend(speed)
        all_dir_changes.extend(dir_change)
        all_accelerations.extend(acceleration)
        all_step_lengths.extend(step_length)
        all_importance.extend(importance)
    
    # Convert to arrays
    all_speeds = np.array(all_speeds)
    all_dir_changes = np.array(all_dir_changes)
    all_accelerations = np.array(all_accelerations)
    all_step_lengths = np.array(all_step_lengths)
    all_importance = np.array(all_importance)
    
    # Compute correlations
    speed_corr, speed_p = stats.pearsonr(all_importance, all_speeds)
    dir_corr, dir_p = stats.pearsonr(all_importance, all_dir_changes)
    accel_corr, accel_p = stats.pearsonr(all_importance, all_accelerations)
    
    print(f"\n🎯 Correlations (Importance vs Feature):")
    print(f"   Speed:           r = {speed_corr:+.3f}  (p = {speed_p:.2e})")
    print(f"   Direction change: r = {dir_corr:+.3f}  (p = {dir_p:.2e})")
    print(f"   Acceleration:    r = {accel_corr:+.3f}  (p = {accel_p:.2e})")
    
    # Create scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Subsample for plotting (too many points = slow)
    plot_n = min(5000, len(all_importance))
    plot_idx = np.random.choice(len(all_importance), plot_n, replace=False)
    
    # Plot 1: Speed vs Importance
    ax1 = axes[0]
    ax1.scatter(all_speeds[plot_idx], all_importance[plot_idx], 
                alpha=0.3, s=10, c='#e74c3c', edgecolors='none')
    # Add trend line
    z = np.polyfit(all_speeds, all_importance, 1)
    p = np.poly1d(z)
    x_line = np.linspace(all_speeds.min(), all_speeds.max(), 100)
    ax1.plot(x_line, p(x_line), 'k-', linewidth=2, label=f'r = {speed_corr:.3f}')
    ax1.set_xlabel('Speed (step length)', fontsize=11)
    ax1.set_ylabel('Importance', fontsize=11)
    ax1.set_title('Speed vs Attention', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Direction Change vs Importance
    ax2 = axes[1]
    ax2.scatter(all_dir_changes[plot_idx], all_importance[plot_idx], 
                alpha=0.3, s=10, c='#9b59b6', edgecolors='none')
    z = np.polyfit(all_dir_changes, all_importance, 1)
    p = np.poly1d(z)
    x_line = np.linspace(all_dir_changes.min(), all_dir_changes.max(), 100)
    ax2.plot(x_line, p(x_line), 'k-', linewidth=2, label=f'r = {dir_corr:.3f}')
    ax2.set_xlabel('Direction Change (rad)', fontsize=11)
    ax2.set_ylabel('Importance', fontsize=11)
    ax2.set_title('Direction Change vs Attention', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Acceleration vs Importance
    ax3 = axes[2]
    ax3.scatter(all_accelerations[plot_idx], all_importance[plot_idx], 
                alpha=0.3, s=10, c='#3498db', edgecolors='none')
    z = np.polyfit(all_accelerations, all_importance, 1)
    p = np.poly1d(z)
    x_line = np.linspace(all_accelerations.min(), all_accelerations.max(), 100)
    ax3.plot(x_line, p(x_line), 'k-', linewidth=2, label=f'r = {accel_corr:.3f}')
    ax3.set_xlabel('Acceleration (Δspeed)', fontsize=11)
    ax3.set_ylabel('Importance', fontsize=11)
    ax3.set_title('Acceleration vs Attention', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format='svg', transparent=True, bbox_inches='tight', dpi=150)
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    
    return {
        'speed_correlation': speed_corr,
        'direction_correlation': dir_corr,
        'acceleration_correlation': accel_corr,
        # Raw data for further analysis
        'importance': all_importance,
        'speeds': all_speeds,
        'dir_changes': all_dir_changes,
        'accelerations': all_accelerations,
    }


def analyze_attention_spikes(
    model,
    dataset,
    device="cuda",
    n_samples=200,
    method="gradient",
    spike_percentile=90,
    save_path=None,
):
    """
    Analyze co-occurrence of attention spikes with motion feature spikes.
    
    Identifies frames where attention is high AND a motion feature (speed, 
    direction change) is also high. This reveals what motion events 
    consistently trigger model attention across many tracks.
    
    Parameters
    ----------
    model : nn.Module
        Trained encoder model
    dataset : Dataset
        Trajectory dataset
    device : str, default="cuda"
        Computation device
    n_samples : int, default=200
        Number of windows to analyze
    method : str, default="gradient"
        Importance method ('gradient' is faster)
    spike_percentile : float, default=90
        Percentile threshold for defining "spikes"
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    dict
        Analysis results with spike co-occurrence statistics
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    print(f"📊 Analyzing attention spike co-occurrence...")
    print(f"   Spike threshold: top {100-spike_percentile:.0f}% of values")
    
    model.eval()
    model.to(device)
    
    # Collect per-window spike data
    window_data = []
    
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    for idx in tqdm(indices, desc="Analyzing windows"):
        sample = dataset[idx]
        
        if 'features' in sample:
            features = sample['features'].unsqueeze(0).to(device)
        else:
            features = sample['features_t'].unsqueeze(0).to(device)
        
        # Get importance
        importance = model.extract_attention_weights(features, method=method)
        importance = importance.squeeze().cpu().numpy()
        
        # Extract motion features
        feat_np = features.squeeze().cpu().numpy()
        dx, dy = feat_np[:, 0], feat_np[:, 1]
        speed = np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx)
        dir_change = np.abs(np.diff(direction, prepend=direction[0]))
        dir_change = np.minimum(dir_change, 2*np.pi - dir_change)
        accel = np.abs(np.diff(speed, prepend=speed[0]))
        
        # Identify spikes (top X% within this window)
        imp_thresh = np.percentile(importance, spike_percentile)
        speed_thresh = np.percentile(speed, spike_percentile)
        dir_thresh = np.percentile(dir_change, spike_percentile)
        accel_thresh = np.percentile(accel, spike_percentile)
        
        imp_spikes = importance > imp_thresh
        speed_spikes = speed > speed_thresh
        dir_spikes = dir_change > dir_thresh
        accel_spikes = accel > accel_thresh
        
        # Count co-occurrences
        n_frames = len(importance)
        n_imp_spikes = imp_spikes.sum()
        
        # How many attention spikes coincide with feature spikes?
        imp_and_speed = (imp_spikes & speed_spikes).sum()
        imp_and_dir = (imp_spikes & dir_spikes).sum()
        imp_and_accel = (imp_spikes & accel_spikes).sum()
        
        # Expected by chance (if independent)
        expected_speed = n_imp_spikes * speed_spikes.sum() / n_frames
        expected_dir = n_imp_spikes * dir_spikes.sum() / n_frames
        expected_accel = n_imp_spikes * accel_spikes.sum() / n_frames
        
        window_data.append({
            'n_imp_spikes': n_imp_spikes,
            'imp_and_speed': imp_and_speed,
            'imp_and_dir': imp_and_dir,
            'imp_and_accel': imp_and_accel,
            'expected_speed': expected_speed,
            'expected_dir': expected_dir,
            'expected_accel': expected_accel,
        })
    
    # Aggregate across windows
    total_imp_spikes = sum(w['n_imp_spikes'] for w in window_data)
    total_imp_speed = sum(w['imp_and_speed'] for w in window_data)
    total_imp_dir = sum(w['imp_and_dir'] for w in window_data)
    total_imp_accel = sum(w['imp_and_accel'] for w in window_data)
    
    total_expected_speed = sum(w['expected_speed'] for w in window_data)
    total_expected_dir = sum(w['expected_dir'] for w in window_data)
    total_expected_accel = sum(w['expected_accel'] for w in window_data)
    
    # Enrichment ratios (observed / expected)
    speed_enrichment = total_imp_speed / (total_expected_speed + 1e-8)
    dir_enrichment = total_imp_dir / (total_expected_dir + 1e-8)
    accel_enrichment = total_imp_accel / (total_expected_accel + 1e-8)
    
    print(f"\n🎯 Attention Spike Co-occurrence:")
    print(f"   Total attention spikes: {total_imp_spikes}")
    print(f"")
    print(f"   Speed spikes:     {total_imp_speed:4d} ({100*total_imp_speed/total_imp_spikes:.1f}%) | "
          f"Expected: {total_expected_speed:.0f} | Enrichment: {speed_enrichment:.2f}x")
    print(f"   Direction spikes: {total_imp_dir:4d} ({100*total_imp_dir/total_imp_spikes:.1f}%) | "
          f"Expected: {total_expected_dir:.0f} | Enrichment: {dir_enrichment:.2f}x")
    print(f"   Accel spikes:     {total_imp_accel:4d} ({100*total_imp_accel/total_imp_spikes:.1f}%) | "
          f"Expected: {total_expected_accel:.0f} | Enrichment: {accel_enrichment:.2f}x")
    print(f"")
    
    # Interpret
    if speed_enrichment > 1.3:
        print(f"   → Attention preferentially fires during FAST MOTION ({speed_enrichment:.1f}x enrichment)")
    if dir_enrichment > 1.3:
        print(f"   → Attention preferentially fires during DIRECTION CHANGES ({dir_enrichment:.1f}x enrichment)")
    if accel_enrichment > 1.3:
        print(f"   → Attention preferentially fires during ACCELERATION events ({accel_enrichment:.1f}x enrichment)")
    
    if max(speed_enrichment, dir_enrichment, accel_enrichment) < 1.2:
        print(f"   → Attention is NOT strongly coupled to any single motion feature")
        print(f"      (may respond to combinations or higher-order patterns)")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    features = ['Speed', 'Direction\nChange', 'Acceleration']
    observed = [total_imp_speed, total_imp_dir, total_imp_accel]
    expected = [total_expected_speed, total_expected_dir, total_expected_accel]
    enrichments = [speed_enrichment, dir_enrichment, accel_enrichment]
    colors = ['#e74c3c', '#9b59b6', '#3498db']
    
    # Bar plot of co-occurrence
    ax1 = axes[0]
    x = np.arange(3)
    width = 0.35
    bars1 = ax1.bar(x - width/2, observed, width, label='Observed', color=colors, edgecolor='black')
    bars2 = ax1.bar(x + width/2, expected, width, label='Expected (random)', 
                    color='gray', alpha=0.5, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(features)
    ax1.set_ylabel('# Attention spikes with feature spike', fontsize=10)
    ax1.set_title('Co-occurrence: Attention + Feature Spikes', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Enrichment bar plot
    ax2 = axes[1]
    bars = ax2.bar(features, enrichments, color=colors, edgecolor='black')
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Random chance')
    ax2.set_ylabel('Enrichment (Observed / Expected)', fontsize=10)
    ax2.set_title('Enrichment of Feature Spikes\nin Attention Spikes', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, val in zip(bars, enrichments):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                 f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Summary pie chart
    ax3 = axes[2]
    # What motion feature best explains attention spikes?
    labels = ['Speed', 'Direction', 'Acceleration', 'Other/Combined']
    # Calculate unique contributions (rough approximation)
    unique_speed = max(0, total_imp_speed - total_expected_speed)
    unique_dir = max(0, total_imp_dir - total_expected_dir) 
    unique_accel = max(0, total_imp_accel - total_expected_accel)
    other = max(0, total_imp_spikes - unique_speed - unique_dir - unique_accel)
    sizes = [unique_speed, unique_dir, unique_accel, other]
    if sum(sizes) > 0:
        colors_pie = ['#e74c3c', '#9b59b6', '#3498db', '#95a5a6']
        ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%', 
                startangle=90, explode=(0.02, 0.02, 0.02, 0))
        ax3.set_title('What Triggers Attention?', fontsize=11, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No clear pattern', ha='center', va='center', fontsize=12)
        ax3.set_title('What Triggers Attention?', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format='svg', transparent=True, bbox_inches='tight', dpi=150)
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    
    return {
        'speed_enrichment': speed_enrichment,
        'direction_enrichment': dir_enrichment,
        'acceleration_enrichment': accel_enrichment,
        'total_attention_spikes': total_imp_spikes,
        'window_data': window_data,
    }


def analyze_attention_with_dataframe_features(
    model,
    dataset,
    instant_df,
    feature_names,
    device="cuda",
    n_samples=500,
    method="gradient",
    save_path=None,
    colors=None,
):
    """
    Analyze attention correlation with PRE-COMPUTED features from a dataframe.
    
    This function uses features that are already calculated and stored in instant_df
    (e.g., from features.py), rather than re-computing them from raw trajectory data.
    
    Parameters
    ----------
    model : nn.Module
        Trained encoder model
    dataset : TimeAwareTrajectoryDataset
        Dataset with trajectory windows (must have pair_indices with row info)
    instant_df : pd.DataFrame or pl.DataFrame
        DataFrame with pre-computed features. Must contain:
        - 'unique_id': Track identifier
        - 'frame': Frame number
        - All columns specified in feature_names
    feature_names : list of str
        List of column names to analyze. Examples:
        - 'speed_um_s': Instantaneous speed
        - 'anomalous_exponent': Alpha from MSD fit (note: this is per-window, not per-frame)
        - 'cum_displacement_um': Cumulative displacement
        - 'self_intersections': Number of self-intersections (per-window feature)
        - 'acceleration_um_s2': Instantaneous acceleration
        - 'direction_rad': Direction of motion
        - 'normalized_curvature': Curvature normalized by distance
    device : str, default="cuda"
        Computation device
    n_samples : int, default=500
        Number of windows to analyze
    method : str, default="gradient"
        Importance method: 'gradient' (fast) or 'leave_one_out' (intuitive)
    save_path : str or Path, optional
        Path to save figure
    colors : list of str, optional
        List of colors for each feature. If None, uses default palette.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'correlations': Dict mapping feature_name -> (correlation, p_value)
        - 'importance': Array of all importance scores
        - 'features': Dict mapping feature_name -> array of feature values
        - 'n_samples_analyzed': Number of windows actually analyzed
        
    Examples
    --------
    >>> # Analyze with pre-computed features from instant_df
    >>> results = analyze_attention_with_dataframe_features(
    ...     model=trainer.model,
    ...     dataset=test_dataset,
    ...     instant_df=instant_df,
    ...     feature_names=['speed_um_s', 'cum_displacement_um', 'acceleration_um_s2'],
    ... )
    >>> for name, (corr, p) in results['correlations'].items():
    ...     print(f"{name}: r={corr:.3f}, p={p:.2e}")
    
    Notes
    -----
    - For per-frame features (speed, acceleration, etc.), values are matched to 
      each frame in the window and correlated with per-frame attention importance.
    - For per-window features (anomalous_exponent, self_intersections), the same
      value is used for all frames in that window (less meaningful for frame-level
      correlation, but still shows overall association).
    - The dataset must be a TimeAwareTrajectoryDataset with access to pair_indices
      and track_index for looking up the correct rows in instant_df.
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    from pathlib import Path
    
    print(f"📊 Analyzing attention vs {len(feature_names)} pre-computed features ({n_samples} samples)...")
    print(f"   Features: {feature_names}")
    
    model.eval()
    model.to(device)
    
    # Convert instant_df to pandas if it's polars
    if hasattr(instant_df, 'to_pandas'):
        df = instant_df.to_pandas()
    else:
        df = instant_df
    
    # Check which features exist in the dataframe
    available_features = []
    missing_features = []
    for feat in feature_names:
        if feat in df.columns:
            available_features.append(feat)
        else:
            missing_features.append(feat)
    
    if missing_features:
        print(f"   ⚠️ Missing features (not in dataframe): {missing_features}")
        print(f"   Available columns: {list(df.columns)[:20]}...")
    
    if not available_features:
        print("   ❌ No valid features found in dataframe!")
        return None
    
    print(f"   ✓ Analyzing features: {available_features}")
    
    # Initialize storage for each feature
    all_importance = []
    all_features = {feat: [] for feat in available_features}
    
    # Create index for fast lookup: (unique_id, frame) -> row
    df_indexed = df.set_index(['unique_id', 'frame'])
    
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    n_analyzed = 0
    for idx in tqdm(indices, desc="Extracting features"):
        try:
            # Get sample from dataset
            sample = dataset[idx]
            unique_id = sample['unique_id']
            
            # Get the window's row range from dataset
            _, row_start_t, _ = dataset.pair_indices[idx]
            window_size = dataset.window_size
            
            # Get features tensor for attention importance
            if 'features' in sample:
                features_tensor = sample['features'].unsqueeze(0).to(device)
            else:
                features_tensor = sample['features_t'].unsqueeze(0).to(device)
            
            # Compute attention importance
            importance = model.extract_attention_weights(features_tensor, method=method)
            importance = importance.squeeze().cpu().numpy()
            
            # Get the frames for this window from the dataset's arrays
            frames_in_window = dataset.frames[row_start_t:row_start_t + window_size]
            
            # Extract pre-computed features for each frame in the window
            window_features = {feat: [] for feat in available_features}
            valid_frames = 0
            
            for frame in frames_in_window:
                try:
                    row = df_indexed.loc[(unique_id, frame)]
                    for feat in available_features:
                        val = row[feat] if feat in row.index else np.nan
                        window_features[feat].append(val)
                    valid_frames += 1
                except KeyError:
                    # Frame not found in dataframe - use NaN
                    for feat in available_features:
                        window_features[feat].append(np.nan)
            
            if valid_frames < window_size * 0.5:
                # Skip if less than half the frames were found
                continue
            
            # Add to collection (only non-NaN values)
            for i, imp in enumerate(importance):
                has_all_features = True
                feature_vals = {}
                for feat in available_features:
                    val = window_features[feat][i] if i < len(window_features[feat]) else np.nan
                    if np.isnan(val) if isinstance(val, float) else False:
                        has_all_features = False
                        break
                    feature_vals[feat] = val
                
                if has_all_features:
                    all_importance.append(imp)
                    for feat in available_features:
                        all_features[feat].append(feature_vals[feat])
            
            n_analyzed += 1
            
        except Exception as e:
            # Skip problematic samples
            continue
    
    print(f"   ✓ Analyzed {n_analyzed} windows, {len(all_importance)} frame-feature pairs")
    
    if len(all_importance) < 10:
        print("   ❌ Not enough valid data points for correlation analysis!")
        return None
    
    # Convert to arrays
    all_importance = np.array(all_importance)
    for feat in available_features:
        all_features[feat] = np.array(all_features[feat])
    
    # Compute correlations
    correlations = {}
    print(f"\n🎯 Correlations (Importance vs Feature):")
    for feat in available_features:
        try:
            corr, p_val = stats.pearsonr(all_importance, all_features[feat])
            correlations[feat] = (corr, p_val)
            print(f"   {feat:<30} r = {corr:+.3f}  (p = {p_val:.2e})")
        except Exception as e:
            print(f"   {feat:<30} ❌ Error: {e}")
            correlations[feat] = (np.nan, np.nan)
    
    # Create scatter plots
    n_features = len(available_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_features > 1 else axes
    
    # Default colors
    default_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#34495e']
    if colors is None:
        colors = default_colors
    
    # Subsample for plotting
    plot_n = min(5000, len(all_importance))
    plot_idx = np.random.choice(len(all_importance), plot_n, replace=False)
    
    for i, feat in enumerate(available_features):
        ax = axes[i] if n_features > 1 else axes[0]
        color = colors[i % len(colors)]
        
        feat_vals = all_features[feat]
        corr, p_val = correlations[feat]
        
        ax.scatter(feat_vals[plot_idx], all_importance[plot_idx], 
                   alpha=0.3, s=10, c=color, edgecolors='none')
        
        # Add trend line
        try:
            z = np.polyfit(feat_vals, all_importance, 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.nanmin(feat_vals), np.nanmax(feat_vals), 100)
            ax.plot(x_line, p(x_line), 'k-', linewidth=2, label=f'r = {corr:.3f}')
        except:
            pass
        
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel('Importance', fontsize=10)
        ax.set_title(f'{feat}\nvs Attention', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes) if hasattr(axes, '__len__') else 1):
        if hasattr(axes, '__len__'):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format='svg', transparent=True, bbox_inches='tight', dpi=150)
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    
    return {
        'correlations': correlations,
        'importance': all_importance,
        'features': all_features,
        'n_samples_analyzed': n_analyzed,
        'feature_names': available_features,
    }


def analyze_attention_spikes_with_features(
    model,
    dataset,
    instant_df,
    feature_names,
    device="cuda",
    n_samples=200,
    method="gradient",
    spike_percentile=90,
    save_path=None,
):
    """
    Analyze co-occurrence of attention spikes with PRE-COMPUTED feature spikes.
    
    Uses features already calculated and stored in instant_df rather than
    re-computing them. Shows whether attention spikes align with spikes in
    the specified features.
    
    Parameters
    ----------
    model : nn.Module
        Trained encoder model
    dataset : TimeAwareTrajectoryDataset
        Dataset with trajectory windows
    instant_df : pd.DataFrame or pl.DataFrame
        DataFrame with pre-computed features
    feature_names : list of str
        List of column names to analyze for spike co-occurrence
    device : str, default="cuda"
        Computation device
    n_samples : int, default=200
        Number of windows to analyze
    method : str, default="gradient"
        Importance method
    spike_percentile : float, default=90
        Percentile threshold for defining "spikes" (top 10% by default)
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'enrichments': Dict mapping feature_name -> enrichment_ratio
        - 'observed_counts': Dict mapping feature_name -> observed co-occurrences
        - 'expected_counts': Dict mapping feature_name -> expected by chance
        - 'total_attention_spikes': Total number of attention spikes
        
    Examples
    --------
    >>> results = analyze_attention_spikes_with_features(
    ...     model=trainer.model,
    ...     dataset=test_dataset,
    ...     instant_df=instant_df,
    ...     feature_names=['speed_um_s', 'acceleration_um_s2', 'normalized_curvature'],
    ...     spike_percentile=85,
    ... )
    >>> for name, enrichment in results['enrichments'].items():
    ...     print(f"{name}: {enrichment:.2f}x expected")
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    print(f"📊 Analyzing attention spike co-occurrence with {len(feature_names)} features...")
    print(f"   Spike threshold: top {100-spike_percentile:.0f}% of values")
    
    model.eval()
    model.to(device)
    
    # Convert instant_df to pandas if needed
    if hasattr(instant_df, 'to_pandas'):
        df = instant_df.to_pandas()
    else:
        df = instant_df
    
    # Check which features exist
    available_features = [f for f in feature_names if f in df.columns]
    if not available_features:
        print(f"   ❌ No valid features found! Available: {list(df.columns)[:10]}...")
        return None
    
    print(f"   ✓ Analyzing: {available_features}")
    
    # Create index for fast lookup
    df_indexed = df.set_index(['unique_id', 'frame'])
    
    # Collect per-window spike data
    window_data = []
    
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    for idx in tqdm(indices, desc="Analyzing windows"):
        try:
            sample = dataset[idx]
            unique_id = sample['unique_id']
            
            _, row_start_t, _ = dataset.pair_indices[idx]
            window_size = dataset.window_size
            
            if 'features' in sample:
                features_tensor = sample['features'].unsqueeze(0).to(device)
            else:
                features_tensor = sample['features_t'].unsqueeze(0).to(device)
            
            # Get importance
            importance = model.extract_attention_weights(features_tensor, method=method)
            importance = importance.squeeze().cpu().numpy()
            
            # Get frames for this window
            frames_in_window = dataset.frames[row_start_t:row_start_t + window_size]
            
            # Extract features for each frame
            window_features = {feat: [] for feat in available_features}
            for frame in frames_in_window:
                try:
                    row = df_indexed.loc[(unique_id, frame)]
                    for feat in available_features:
                        val = row[feat] if feat in row.index else np.nan
                        window_features[feat].append(val)
                except KeyError:
                    for feat in available_features:
                        window_features[feat].append(np.nan)
            
            # Convert to arrays and remove NaN
            for feat in available_features:
                window_features[feat] = np.array(window_features[feat])
            
            # Identify spikes (top X% within this window)
            imp_thresh = np.percentile(importance, spike_percentile)
            imp_spikes = importance > imp_thresh
            
            n_frames = len(importance)
            n_imp_spikes = imp_spikes.sum()
            
            # Count co-occurrences for each feature
            window_stats = {
                'n_imp_spikes': n_imp_spikes,
                'n_frames': n_frames,
            }
            
            for feat in available_features:
                feat_vals = window_features[feat]
                valid_mask = ~np.isnan(feat_vals)
                
                if valid_mask.sum() > 0:
                    feat_thresh = np.nanpercentile(feat_vals, spike_percentile)
                    feat_spikes = feat_vals > feat_thresh
                    
                    # Co-occurrence
                    co_occur = (imp_spikes & feat_spikes & valid_mask).sum()
                    expected = n_imp_spikes * feat_spikes.sum() / n_frames if n_frames > 0 else 0
                    
                    window_stats[f'{feat}_co'] = co_occur
                    window_stats[f'{feat}_exp'] = expected
                else:
                    window_stats[f'{feat}_co'] = 0
                    window_stats[f'{feat}_exp'] = 0
            
            window_data.append(window_stats)
            
        except Exception as e:
            continue
    
    if not window_data:
        print("   ❌ No valid windows analyzed!")
        return None
    
    # Aggregate across windows
    total_imp_spikes = sum(w['n_imp_spikes'] for w in window_data)
    
    enrichments = {}
    observed_counts = {}
    expected_counts = {}
    
    print(f"\n🎯 Attention Spike Co-occurrence:")
    print(f"   Total attention spikes: {total_imp_spikes}")
    print()
    
    for feat in available_features:
        total_co = sum(w.get(f'{feat}_co', 0) for w in window_data)
        total_exp = sum(w.get(f'{feat}_exp', 0) for w in window_data)
        
        enrichment = total_co / (total_exp + 1e-8)
        
        enrichments[feat] = enrichment
        observed_counts[feat] = total_co
        expected_counts[feat] = total_exp
        
        pct = (total_co / total_imp_spikes * 100) if total_imp_spikes > 0 else 0
        print(f"   {feat:<30} {total_co:4d} ({pct:.1f}%) | Expected: {total_exp:.0f} | Enrichment: {enrichment:.2f}x")
    
    # Interpretation
    print()
    for feat, enrichment in enrichments.items():
        if enrichment > 1.3:
            print(f"   → Attention preferentially fires during high {feat} ({enrichment:.1f}x enrichment)")
    
    max_enrichment = max(enrichments.values()) if enrichments else 0
    if max_enrichment < 1.2:
        print(f"   → Attention is NOT strongly coupled to any single feature")
    
    # Create visualization
    n_features = len(available_features)
    fig, axes = plt.subplots(1, min(3, n_features + 1), figsize=(14, 4))
    if n_features + 1 <= 3:
        axes = [axes] if n_features + 1 == 1 else list(axes)
    
    # Default colors
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#34495e']
    
    # Bar plot of co-occurrence
    ax1 = axes[0]
    x = np.arange(n_features)
    width = 0.35
    observed = [observed_counts[f] for f in available_features]
    expected = [expected_counts[f] for f in available_features]
    
    bars1 = ax1.bar(x - width/2, observed, width, label='Observed', 
                    color=[colors[i % len(colors)] for i in range(n_features)], edgecolor='black')
    bars2 = ax1.bar(x + width/2, expected, width, label='Expected (random)', 
                    color='gray', alpha=0.5, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f[:15] for f in available_features], rotation=45, ha='right')
    ax1.set_ylabel('# Co-occurrences')
    ax1.set_title('Attention + Feature Spikes', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Enrichment bar plot
    if len(axes) > 1:
        ax2 = axes[1]
        enrichment_vals = [enrichments[f] for f in available_features]
        bars = ax2.bar(range(n_features), enrichment_vals, 
                       color=[colors[i % len(colors)] for i in range(n_features)], edgecolor='black')
        ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Random chance')
        ax2.set_xticks(range(n_features))
        ax2.set_xticklabels([f[:15] for f in available_features], rotation=45, ha='right')
        ax2.set_ylabel('Enrichment (Obs / Exp)')
        ax2.set_title('Enrichment Ratios', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, enrichment_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                     f'{val:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Hide extra axes
    for i in range(2, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format='svg', transparent=True, bbox_inches='tight', dpi=150)
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    
    return {
        'enrichments': enrichments,
        'observed_counts': observed_counts,
        'expected_counts': expected_counts,
        'total_attention_spikes': total_imp_spikes,
        'feature_names': available_features,
        'n_windows': len(window_data),
    }


def analyze_attention_with_window_features(
    model,
    dataset,
    time_windowed_df,
    feature_names,
    device="cuda",
    n_samples=500,
    method="gradient",
    aggregation='mean',
    save_path=None,
    colors=None,
):
    """
    Analyze attention correlation with WINDOW-LEVEL features from time_windowed_df.
    
    For features that are calculated per-window (not per-frame) like anomalous_exponent,
    self_intersections, D_fit, etc. This aggregates frame-level attention importance
    (mean, max, etc.) and correlates with window-level features.
    
    Parameters
    ----------
    model : nn.Module
        Trained encoder model
    dataset : TimeAwareTrajectoryDataset
        Dataset with trajectory windows
    time_windowed_df : pd.DataFrame
        DataFrame with pre-computed WINDOW-LEVEL features. Must contain:
        - 'window_uid' or ('unique_id' + 'time_window'): Window identifier
        - All columns specified in feature_names
    feature_names : list of str
        List of window-level feature column names. Examples:
        - 'anomalous_exponent' or 'alpha': From MSD fit
        - 'D_fit': Diffusion coefficient from fit
        - 'self_intersections': Number of trajectory self-crossings
        - 'straightness_index': Straightness of trajectory
        - 'radius_of_gyration': Spatial extent
        - 'convex_hull_area': Area covered by trajectory
        - 'fractal_dimension': Complexity measure
    device : str, default="cuda"
        Computation device
    n_samples : int, default=500
        Number of windows to analyze
    method : str, default="gradient"
        Importance method
    aggregation : str, default='mean'
        How to aggregate frame-level importance: 'mean', 'max', 'std', 'sum'
    save_path : str or Path, optional
        Path to save figure
    colors : list of str, optional
        Colors for each feature plot
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'correlations': Dict mapping feature_name -> (correlation, p_value)
        - 'window_importance': Array of aggregated importance per window
        - 'features': Dict mapping feature_name -> array of feature values
        
    Examples
    --------
    >>> # Analyze with window-level features
    >>> results = analyze_attention_with_window_features(
    ...     model=trainer.model,
    ...     dataset=test_dataset,
    ...     time_windowed_df=time_windowed_df,
    ...     feature_names=['anomalous_exponent', 'self_intersections', 'D_fit'],
    ... )
    >>> for name, (corr, p) in results['correlations'].items():
    ...     print(f"{name}: r={corr:.3f}")
    
    Notes
    -----
    This function correlates WINDOW-LEVEL metrics (one value per window) with
    aggregated attention importance (one value per window). This is appropriate
    for features like anomalous_exponent that don't have per-frame values.
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    from pathlib import Path
    
    print(f"📊 Analyzing attention vs {len(feature_names)} window-level features ({n_samples} samples)...")
    print(f"   Attention aggregation: {aggregation}")
    print(f"   Features: {feature_names}")
    
    model.eval()
    model.to(device)
    
    # Convert to pandas if needed
    if hasattr(time_windowed_df, 'to_pandas'):
        wdf = time_windowed_df.to_pandas()
    else:
        wdf = time_windowed_df
    
    # Check which features exist
    available_features = [f for f in feature_names if f in wdf.columns]
    missing_features = [f for f in feature_names if f not in wdf.columns]
    
    if missing_features:
        print(f"   ⚠️ Missing features: {missing_features}")
    
    if not available_features:
        print(f"   ❌ No valid features! Available: {list(wdf.columns)[:15]}...")
        return None
    
    print(f"   ✓ Analyzing: {available_features}")
    
    # Create lookup for window features
    # Try window_uid first, then unique_id + time_window
    if 'window_uid' in wdf.columns:
        wdf_indexed = wdf.set_index('window_uid')
        use_window_uid = True
    else:
        wdf_indexed = wdf.set_index(['unique_id', 'time_window'])
        use_window_uid = False
    
    # Storage
    all_window_importance = []
    all_features = {feat: [] for feat in available_features}
    
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    n_analyzed = 0
    for idx in tqdm(indices, desc="Analyzing windows"):
        try:
            sample = dataset[idx]
            unique_id = sample['unique_id']
            time_window = sample.get('time_window_t', sample.get('time_window', 0))
            
            # Get features tensor
            if 'features' in sample:
                features_tensor = sample['features'].unsqueeze(0).to(device)
            else:
                features_tensor = sample['features_t'].unsqueeze(0).to(device)
            
            # Compute attention importance
            importance = model.extract_attention_weights(features_tensor, method=method)
            importance = importance.squeeze().cpu().numpy()
            
            # Aggregate importance for this window
            if aggregation == 'mean':
                window_imp = np.mean(importance)
            elif aggregation == 'max':
                window_imp = np.max(importance)
            elif aggregation == 'std':
                window_imp = np.std(importance)
            elif aggregation == 'sum':
                window_imp = np.sum(importance)
            else:
                window_imp = np.mean(importance)
            
            # Look up window features
            try:
                if use_window_uid:
                    # Try to construct window_uid
                    _, row_start_t, _ = dataset.pair_indices[idx]
                    start_frame = dataset.frames[row_start_t]
                    end_frame = dataset.frames[row_start_t + dataset.window_size - 1]
                    window_uid = f"{unique_id}_{time_window}_{start_frame}_{end_frame}"
                    
                    if window_uid in wdf_indexed.index:
                        row = wdf_indexed.loc[window_uid]
                    else:
                        # Try without exact frame numbers (partial match)
                        matches = [uid for uid in wdf_indexed.index if uid.startswith(f"{unique_id}_{time_window}_")]
                        if matches:
                            row = wdf_indexed.loc[matches[0]]
                        else:
                            continue
                else:
                    row = wdf_indexed.loc[(unique_id, time_window)]
                
                # Extract features
                feature_vals = {}
                all_valid = True
                for feat in available_features:
                    val = row[feat] if feat in row.index else np.nan
                    if pd.isna(val):
                        all_valid = False
                        break
                    feature_vals[feat] = val
                
                if all_valid:
                    all_window_importance.append(window_imp)
                    for feat in available_features:
                        all_features[feat].append(feature_vals[feat])
                    n_analyzed += 1
                    
            except (KeyError, IndexError):
                continue
                
        except Exception as e:
            continue
    
    print(f"   ✓ Analyzed {n_analyzed} windows")
    
    if n_analyzed < 10:
        print("   ❌ Not enough valid data for correlation analysis!")
        return None
    
    # Convert to arrays
    all_window_importance = np.array(all_window_importance)
    for feat in available_features:
        all_features[feat] = np.array(all_features[feat])
    
    # Compute correlations
    correlations = {}
    print(f"\n🎯 Window-Level Correlations (Aggregated Attention vs Feature):")
    for feat in available_features:
        try:
            corr, p_val = stats.pearsonr(all_window_importance, all_features[feat])
            correlations[feat] = (corr, p_val)
            print(f"   {feat:<30} r = {corr:+.3f}  (p = {p_val:.2e})")
        except Exception as e:
            print(f"   {feat:<30} ❌ Error: {e}")
            correlations[feat] = (np.nan, np.nan)
    
    # Create scatter plots
    n_features = len(available_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_features > 1 else [axes]
    
    default_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#34495e']
    if colors is None:
        colors = default_colors
    
    for i, feat in enumerate(available_features):
        ax = axes[i]
        color = colors[i % len(colors)]
        
        feat_vals = all_features[feat]
        corr, p_val = correlations[feat]
        
        ax.scatter(feat_vals, all_window_importance, alpha=0.5, s=20, c=color, edgecolors='none')
        
        # Add trend line
        try:
            z = np.polyfit(feat_vals, all_window_importance, 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.nanmin(feat_vals), np.nanmax(feat_vals), 100)
            ax.plot(x_line, p(x_line), 'k-', linewidth=2, label=f'r = {corr:.3f}')
        except:
            pass
        
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel(f'{aggregation.capitalize()} Attention', fontsize=10)
        ax.set_title(f'{feat}\nvs Window Attention', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format='svg', transparent=True, bbox_inches='tight', dpi=150)
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    
    return {
        'correlations': correlations,
        'window_importance': all_window_importance,
        'features': all_features,
        'n_samples_analyzed': n_analyzed,
        'feature_names': available_features,
        'aggregation': aggregation,
    }


# Convenience functions for easy use from notebooks
def create_trajectory_dataset(
    instant_df, window_size=60, overlap=30, min_track_length=60
):
    """Create a trajectory dataset from pandas DataFrame"""
    return PandasTrajectoryDataset(instant_df, window_size, overlap, min_track_length)


def create_smart_train_val_test_split(
    instant_df,
    val_split=0.15,
    test_split=0.2,
    split_strategy="hybrid_cell",
    random_seed=42,
    balancing_features=None,
    cells_per_condition=8,
    test_condition_col=None,
    # Save options
    save_dir=None,
    save_train=False,
    save_val=False,
    save_test=False,
):
    """
    Create smart train/validation/test splits with multiple strategies.
    
    Supports both Polars and Pandas DataFrames natively, using Polars operations
    when a Polars DataFrame is provided for better performance.

    Parameters
    ----------
    instant_df : pl.DataFrame or pd.DataFrame
        DataFrame with trajectory data
    val_split : float, default=0.15
        Fraction for validation set
    test_split : float, default=0.2
        Fraction for test set
    split_strategy : str, default="hybrid_cell"
        Strategy for splitting data. Options:
        
        - **"fixed_cells"**: Select a fixed number of complete cells per condition 
          for test set. Best for ensuring biological replicates in test.
          Uses `cells_per_condition` and `test_condition_col`.
          
        - **"hybrid_cell"**: Test set contains complete cells (for clean visualization),
          while train/val can have mixed tracks from remaining cells. Proportionally
          balances conditions in test set.
          
        - **"enhanced_hybrid_cell"**: Like hybrid_cell but uses `balancing_features`
          to create more granular condition labels for stratification.
          
        - **"stratified"**: Stratified split by condition at the track level.
          Tracks from the same cell may appear in different splits.
          
        - **"cell_balanced"**: Distributes entire cells across splits.
          Similar to hybrid but applies to all splits, not just test.
          
        - **"random"**: Simple random split at the track level.
          No condition balancing, tracks from same cell may be split.
          
    random_seed : int, default=42
        Random seed for reproducibility
    balancing_features : list of str, optional
        Column names used to create a combined "class_balance_label" for 
        stratified train/val splitting. For example, `["mol"]` creates labels
        based only on molecule type, while `["cell", "type", "mol"]` creates
        more granular labels like "Cell1_TypeA_EGFR".
        
        This ensures train/val splits have balanced representation across
        the specified features. If None, uses default from config or
        falls back to ["cell", "type", "mol"].
        
    cells_per_condition : int, default=8
        Number of cells per condition for test set (used with 'fixed_cells' strategy)
    test_condition_col : str, optional
        Column name to use for test cell selection (if None, uses 'condition').
        For 'fixed_cells' strategy, this determines what "groups" test cells
        are selected from.
    save_dir : str or Path, optional
        Directory to save dataframes as parquet files
    save_train : bool, default=False
        Whether to save train dataframe to parquet
    save_val : bool, default=False
        Whether to save validation dataframe to parquet
    save_test : bool, default=False
        Whether to save test dataframe to parquet

    Returns
    -------
    train_df : pl.DataFrame or pd.DataFrame
        Training data (same type as input)
    val_df : pl.DataFrame or pd.DataFrame
        Validation data
    test_df : pl.DataFrame or pd.DataFrame
        Test data
    split_info : dict
        Dictionary with split statistics and metadata
        
    Examples
    --------
    >>> # Fixed cells strategy: 6 complete cells per molecule type in test
    >>> train_df, val_df, test_df, info = create_smart_train_val_test_split(
    ...     instant_df,
    ...     split_strategy="fixed_cells",
    ...     test_condition_col="mol",
    ...     cells_per_condition=6,
    ...     balancing_features=["mol"],
    ...     save_dir="./splits",
    ...     save_test=True
    ... )
    """
    from pathlib import Path
    
    np.random.seed(random_seed)
    
    # Handle legacy parameter name
    condition_factors = balancing_features

    # Handle both Polars and Pandas DataFrames natively
    is_polars = isinstance(instant_df, pl.DataFrame)
    
    # Handle condition_factors configuration
    if condition_factors is None:
        try:
            from . import config
            condition_factors = config.ANALYSIS_PARAMS["split_params"]["condition_factors"]
        except (ImportError, KeyError):
            # Fallback to default
            condition_factors = ["cell", "type", "mol"]
    
    # Create class balance label column if using enhanced_hybrid_cell or fixed_cells strategy
    if split_strategy in ["enhanced_hybrid_cell", "fixed_cells"]:
        print(f"🔧 Creating class balance labels using factors: {condition_factors}")
        
        if is_polars:
            # Create class balance label using Polars
            condition_expr = pl.concat_str([pl.col(factor) for factor in condition_factors], separator="_")
            working_df = instant_df.with_columns(condition_expr.alias("class_balance_label"))
        else:
            # Create class balance label using Pandas
            working_df = instant_df.copy()
            working_df["class_balance_label"] = working_df[condition_factors].apply(
                lambda row: "_".join(row.astype(str)), axis=1
            )
        
        print(f"   ✅ Created class balance labels (sample): {working_df['class_balance_label'].unique()[:5].to_list() if is_polars else working_df['class_balance_label'].unique()[:5].tolist()}")
    else:
        # Use existing condition column
        working_df = instant_df
    
    # Always use original condition column for track info
    condition_col = "condition"
    
    if is_polars:
        # Get track and cell information using Polars
        track_info = (
            working_df
            .group_by("unique_id")
            .agg([
                pl.col(condition_col).first().alias("condition"),
                pl.col("filename").first().alias("filename"),
                pl.col("frame").count().alias("track_length")
            ])
            .to_pandas()  # Convert only this summary to pandas for processing logic
        )
    else:
        # Get track and cell information using Pandas
        track_info = (
            working_df.groupby("unique_id")
            .agg(
                {
                    condition_col: "first",
                    "filename": "first",  # Cell identifier
                    "frame": "count",  # Track length
                }
            )
            .reset_index()
        )
        track_info.columns = ["unique_id", "condition", "filename", "track_length"]

    print("📊 Data Overview:")
    print(f"   Total tracks: {len(track_info)}")
    print(f"   Unique conditions: {track_info['condition'].nunique()}")
    print(f"   Unique cells (filenames): {track_info['filename'].nunique()}")
    print(f"   Strategy: {split_strategy}")

    if split_strategy == "fixed_cells":
        # 🎯 FIXED CELLS STRATEGY: Fixed number of cells per condition in test set
        print(f"\n🎯 Fixed Cells Strategy:")
        print(f"   - Test set: {cells_per_condition} complete cells per condition")
        print("   - Train/Val: Remaining data split by tracks (not cells)")
        
        # Use specified condition column for test cell selection
        test_condition_col = test_condition_col or "condition"
        
        # Get all available conditions from test condition column
        if is_polars:
            all_conditions = working_df[test_condition_col].unique().to_list()
        else:
            all_conditions = working_df[test_condition_col].unique().tolist()
        
        print(f"\n🔍 Selecting {cells_per_condition} cells per condition...")
        print(f"   Available conditions: {all_conditions}")
        
        # Group cells by test condition column
        if is_polars:
            cells_by_condition = (
                working_df.group_by(["filename", test_condition_col])
                .agg([
                    pl.col("unique_id").n_unique().alias("n_tracks"),
                    pl.len().alias("total_frames")
                ])
                .to_pandas()
            )
        else:
            cells_by_condition = (
                working_df.groupby(["filename", test_condition_col])
                .agg({"unique_id": "nunique", "frame": "count"})
                .reset_index()
            )
            cells_by_condition.columns = ["filename", test_condition_col, "n_tracks", "total_frames"]
        
        # Select cells for test set
        test_cells = []
        cells_selected_per_condition = {}
        
        for condition in all_conditions:
            condition_cells = cells_by_condition[
                cells_by_condition[test_condition_col] == condition
            ].copy()
            
            # Sort by number of tracks (prefer cells with more tracks)
            condition_cells = condition_cells.sort_values("n_tracks", ascending=False)
            
            # Select up to cells_per_condition cells
            selected_cells = condition_cells.head(cells_per_condition)
            selected_filenames = selected_cells["filename"].tolist()
            
            test_cells.extend(selected_filenames)
            cells_selected_per_condition[condition] = len(selected_cells)
            
            print(f"   ✅ {condition}: Selected {len(selected_cells)} cells")
            for _, cell in selected_cells.iterrows():
                print(f"      - {cell['filename']}: {cell['n_tracks']} tracks")
        
        print(f"\n📊 Test Set Summary:")
        if is_polars:
            # Create a nice summary table using Polars
            summary_data = []
            for condition, count in cells_selected_per_condition.items():
                summary_data.append({
                    "condition": condition,
                    "cells_selected": count,
                    "target": cells_per_condition,
                    "status": "✅ Complete" if count == cells_per_condition else f"⚠️ Only {count}"
                })
            summary_df = pl.DataFrame(summary_data)
            print(summary_df)
        else:
            for condition, count in cells_selected_per_condition.items():
                status = "✅ Complete" if count == cells_per_condition else f"⚠️ Only {count}"
                print(f"   {condition}: {count}/{cells_per_condition} cells {status}")
        
        # Get test tracks (complete selected cells)
        if is_polars:
            test_df = working_df.filter(pl.col("filename").is_in(test_cells))
            remaining_df = working_df.filter(~pl.col("filename").is_in(test_cells))
        else:
            test_df = working_df[working_df["filename"].isin(test_cells)].copy()
            remaining_df = working_df[~working_df["filename"].isin(test_cells)].copy()
        
        # Split remaining data by tracks (not cells) for train/val using class_balance_label
        if is_polars:
            remaining_track_info = (
                remaining_df.group_by("unique_id")
                .agg([
                    pl.col("class_balance_label").first().alias("condition"),
                    pl.col("filename").first().alias("filename"),
                    pl.col("frame").count().alias("track_length")
                ])
                .to_pandas()
            )
        else:
            remaining_track_info = (
                remaining_df.groupby("unique_id")
                .agg({
                    "class_balance_label": "first",
                    "filename": "first",
                    "frame": "count",
                })
                .reset_index()
            )
            remaining_track_info.columns = ["unique_id", "condition", "filename", "track_length"]
        
        # Stratified split of remaining tracks for train/val using class_balance_label
        train_tracks, val_tracks = _stratified_track_split(
            remaining_track_info, val_split, random_seed
        )
        
        # Extract test track IDs from the test set
        if is_polars:
            test_track_ids = test_df["unique_id"].unique().to_list()
        else:
            test_track_ids = test_df["unique_id"].unique().tolist()
        
        # Create final dataframes
        if is_polars:
            train_df = remaining_df.filter(pl.col("unique_id").is_in(train_tracks))
            val_df = remaining_df.filter(pl.col("unique_id").is_in(val_tracks))
        else:
            train_df = remaining_df[remaining_df["unique_id"].isin(train_tracks)].copy()
            val_df = remaining_df[remaining_df["unique_id"].isin(val_tracks)].copy()
        
        split_info = {
            "strategy": "fixed_cells",
            "test_cells": test_cells,
            "test_tracks": len(test_track_ids),
            "train_tracks": len(train_tracks),
            "val_tracks": len(val_tracks),
            "test_complete_cells": True,
            "cells_selected_per_condition": cells_selected_per_condition,
            "cells_per_condition": cells_per_condition,
            "test_condition_col": test_condition_col,
        }
        
    elif split_strategy in ["hybrid_cell", "enhanced_hybrid_cell"]:
        # 🎯 HYBRID STRATEGY: Mixed train/val, complete cells for test
        print("\n🎯 Hybrid Cell Strategy:")
        print("   - Test set: Complete cells (clean visualization)")
        print("   - Train/Val: Mixed tracks from remaining cells (robust training)")

        # Group tracks by cell (filename) and condition
        cells_info = (
            track_info.groupby("filename")
            .agg(
                {
                    "unique_id": "count",
                    "condition": lambda x: list(x.unique()),
                    "track_length": "sum",
                }
            )
            .reset_index()
        )
        cells_info.columns = ["filename", "n_tracks", "conditions", "total_frames"]

        # 🔥 NEW: Condition-aware test cell selection
        print("\n🔍 Analyzing cells by condition for balanced test selection...")

        # Get unique conditions and their distribution
        all_conditions = track_info["condition"].unique()
        condition_track_counts = track_info["condition"].value_counts()

        print(f"   Available conditions: {list(all_conditions)}")
        print(f"   Track distribution: {dict(condition_track_counts)}")

        # Group cells by their primary condition (handle multi-condition cells)
        cells_by_condition = {}
        for _, cell in cells_info.iterrows():
            # Handle cells that might have multiple conditions (should be rare)
            primary_condition = (
                cell["conditions"][0] if cell["conditions"] else "unknown"
            )
            if len(cell["conditions"]) > 1:
                print(
                    f"   ⚠️ Cell {cell['filename']} has multiple conditions: {cell['conditions']}"
                )
                print(f"      Using primary condition: {primary_condition}")

            if primary_condition not in cells_by_condition:
                cells_by_condition[primary_condition] = []
            cells_by_condition[primary_condition].append(cell)

        # Calculate target test tracks per condition (proportional to condition frequency)
        target_test_tracks = int(len(track_info) * test_split)
        target_per_condition = {}

        for condition in all_conditions:
            condition_proportion = condition_track_counts[condition] / len(track_info)
            target_per_condition[condition] = int(
                target_test_tracks * condition_proportion
            )

        print(f"   Target test tracks per condition: {target_per_condition}")

        # Select test cells while balancing conditions
        test_cells = []
        selected_per_condition = {condition: 0 for condition in all_conditions}

        # Sort cells within each condition by total frames (prefer larger cells for cleaner visualization)
        for condition in all_conditions:
            if condition in cells_by_condition:
                condition_cells = sorted(
                    cells_by_condition[condition],
                    key=lambda x: x["total_frames"],
                    reverse=True,
                )

                # Select cells for this condition
                for cell in condition_cells:
                    current_selected = selected_per_condition[condition]
                    target_for_condition = target_per_condition[condition]

                    # Add cell if we haven't reached the target for this condition
                    # Or if no cells selected yet for this condition (ensure each condition gets at least one cell)
                    if current_selected < target_for_condition or (
                        current_selected == 0 and len(test_cells) < len(all_conditions)
                    ):
                        test_cells.append(cell["filename"])
                        selected_per_condition[condition] += cell["n_tracks"]

                        print(
                            f"   ✅ Selected cell {cell['filename']} ({condition}): {cell['n_tracks']} tracks"
                        )

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
        if (
            total_selected < target_test_tracks * 0.8
        ):  # If we're significantly under target
            print(
                f"   ⚠️ Only selected {total_selected} tracks, target was {target_test_tracks}"
            )
            print("   📋 Adding additional cells to reach target...")

            # Get all remaining cells
            all_remaining_cells = []
            for condition, cells in cells_by_condition.items():
                for cell in cells:
                    if cell["filename"] not in test_cells:
                        all_remaining_cells.append(cell)

            # Sort by size and add until we reach target
            all_remaining_cells.sort(key=lambda x: x["total_frames"], reverse=True)
            for cell in all_remaining_cells:
                if total_selected >= target_test_tracks:
                    break
                test_cells.append(cell["filename"])
                # Find which condition this cell belongs to
                cell_condition = cell["conditions"][0]
                selected_per_condition[cell_condition] += cell["n_tracks"]
                total_selected += cell["n_tracks"]
                print(
                    f"   📋 Added {cell['filename']} ({cell_condition}): {cell['n_tracks']} tracks"
                )

        # Report final test set composition
        print("\n📊 Final Test Set Composition:")
        for condition in all_conditions:
            selected = selected_per_condition[condition]
            total_condition = condition_track_counts[condition]
            percentage = (
                (selected / total_condition * 100) if total_condition > 0 else 0
            )
            print(
                f"   {condition}: {selected}/{total_condition} tracks ({percentage:.1f}%)"
            )

        # Get test tracks (complete cells)
        test_track_ids = track_info[track_info["filename"].isin(test_cells)][
            "unique_id"
        ].tolist()

        # Remaining tracks for train/val (mixed cells)
        remaining_track_info = track_info[~track_info["filename"].isin(test_cells)]

        # Stratified split of remaining tracks for train/val
        train_tracks, val_tracks = _stratified_track_split(
            remaining_track_info,
            val_fraction=val_split / (1 - test_split),  # Adjust for reduced pool
            random_seed=random_seed,
        )

        split_info = {
            "strategy": "hybrid_cell",
            "test_cells": test_cells,
            "test_tracks": len(test_track_ids),
            "train_tracks": len(train_tracks),
            "val_tracks": len(val_tracks),
            "test_complete_cells": True,
            "test_condition_balance": selected_per_condition,  # NEW: Track condition balance
            "target_per_condition": target_per_condition,  # NEW: Track targets
        }

    elif split_strategy == "stratified":
        # Stratified by condition only
        train_tracks, remaining_tracks = _stratified_track_split(
            track_info, val_fraction=val_split + test_split, random_seed=random_seed
        )

        remaining_info = track_info[track_info["unique_id"].isin(remaining_tracks)]
        val_tracks, test_track_ids = _stratified_track_split(
            remaining_info,
            val_fraction=val_split / (val_split + test_split),
            random_seed=random_seed + 1,
        )

        split_info = {"strategy": "stratified", "test_complete_cells": False}

    elif split_strategy == "cell_balanced":
        # Balance cells across splits
        cells_info = (
            track_info.groupby("filename")
            .agg({"unique_id": list, "condition": lambda x: list(x.unique())})
            .reset_index()
        )

        # Distribute cells across splits
        np.random.shuffle(cells_info.values)
        n_cells = len(cells_info)

        test_cells_end = int(n_cells * test_split)
        val_cells_end = test_cells_end + int(n_cells * val_split)

        test_cells = cells_info.iloc[:test_cells_end]
        val_cells = cells_info.iloc[test_cells_end:val_cells_end]
        train_cells = cells_info.iloc[val_cells_end:]

        test_track_ids = [
            track for tracks in test_cells["unique_id"] for track in tracks
        ]
        val_tracks = [track for tracks in val_cells["unique_id"] for track in tracks]
        train_tracks = [
            track for tracks in train_cells["unique_id"] for track in tracks
        ]

        split_info = {"strategy": "cell_balanced", "test_complete_cells": True}

    else:  # random
        # Simple random split
        all_tracks = track_info["unique_id"].tolist()
        np.random.shuffle(all_tracks)

        n_test = int(len(all_tracks) * test_split)
        n_val = int(len(all_tracks) * val_split)

        test_track_ids = all_tracks[:n_test]
        val_tracks = all_tracks[n_test : n_test + n_val]
        train_tracks = all_tracks[n_test + n_val :]

        split_info = {"strategy": "random", "test_complete_cells": False}

    # Create the split dataframes using native operations
    if is_polars:
        # Use Polars native filtering
        train_df = working_df.filter(pl.col("unique_id").is_in(train_tracks))
        val_df = working_df.filter(pl.col("unique_id").is_in(val_tracks))
        test_df = working_df.filter(pl.col("unique_id").is_in(test_track_ids))
    else:
        # Use Pandas filtering
        train_df = working_df[working_df["unique_id"].isin(train_tracks)].copy()
        val_df = working_df[working_df["unique_id"].isin(val_tracks)].copy()
        test_df = working_df[working_df["unique_id"].isin(test_track_ids)].copy()

    # Update split info with final statistics
    split_info.update(
        {
            "total_tracks": len(track_info),
            "train_tracks": len(train_tracks),
            "val_tracks": len(val_tracks),
            "test_tracks": len(test_track_ids),
            "train_points": len(train_df),
            "val_points": len(val_df),
            "test_points": len(test_df),
        }
    )

    # Report split quality
    print("\n✅ Split Results:")
    print(f"   Train: {len(train_tracks)} tracks ({len(train_df):,} points)")
    print(f"   Val:   {len(val_tracks)} tracks ({len(val_df):,} points)")
    print(f"   Test:  {len(test_track_ids)} tracks ({len(test_df):,} points)")

    # Check condition balance
    print("\n🔍 Condition Balance:")
    # Use the appropriate condition column based on split strategy
    if split_strategy == "enhanced_hybrid_cell":
        condition_col_to_use = "class_balance_label"
    elif split_strategy == "fixed_cells":
        # For fixed_cells, use class_balance_label for reporting if available, otherwise condition
        condition_col_to_use = "class_balance_label" if "class_balance_label" in train_df.columns else "condition"
    else:
        condition_col_to_use = "condition"
    
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(df) > 0:
            if is_polars:
                condition_counts = (
                    df.group_by("unique_id")
                    .agg(pl.col(condition_col_to_use).first())
                    .group_by(condition_col_to_use)
                    .agg(pl.len().alias("count"))
                    .to_pandas()
                    .set_index(condition_col_to_use)["count"]
                )
            else:
                condition_counts = (
                    df.groupby("unique_id")[condition_col_to_use].first().value_counts()
                )
            print(f"   {split_name}: {dict(condition_counts)}")

    if split_strategy in ["hybrid_cell", "enhanced_hybrid_cell", "fixed_cells"]:
        print("\n🎯 Test Set Cell Info:")
        if is_polars:
            test_cell_info = (
                test_df.group_by("filename")
                .agg([
                    pl.col("unique_id").n_unique().alias("n_tracks"),
                    pl.col("condition").first().alias("original_condition"),
                    pl.col(condition_col_to_use).first().alias("balance_label") if condition_col_to_use in test_df.columns else pl.col("condition").first().alias("balance_label")
                ])
                .sort("n_tracks", descending=True)
            )
            
            print("📊 Test Set Cells Summary:")
            print(test_cell_info)
        else:
            test_cell_info = test_df.groupby("filename").agg({
                "unique_id": "nunique", 
                "condition": lambda x: list(x.unique())[0],
                condition_col_to_use: lambda x: list(x.unique())[0] if condition_col_to_use in test_df.columns else None
            }).sort_values("unique_id", ascending=False)
            
            print("📊 Test Set Cells Summary:")
            for filename, info in test_cell_info.iterrows():
                balance_info = f", {info[condition_col_to_use]}" if condition_col_to_use in info and pd.notna(info[condition_col_to_use]) else ""
                print(f"   {filename}: {info['unique_id']} tracks, {info['condition']}{balance_info}")

        # NEW: Show condition balance quality
        if "test_condition_balance" in split_info:
            print("\n🎯 Test Set Condition Balance Quality:")
            actual_balance = split_info["test_condition_balance"]
            target_balance = split_info["target_per_condition"]

            for condition in actual_balance.keys():
                actual = actual_balance[condition]
                target = target_balance[condition]
                ratio = (actual / target) if target > 0 else float("inf")

                if 0.8 <= ratio <= 1.2:  # Within 20% of target
                    status = "✅ Good"
                elif 0.5 <= ratio <= 2.0:  # Within 50% of target
                    status = "⚠️ Acceptable"
                else:
                    status = "❌ Imbalanced"

                print(
                    f"   {condition}: {actual}/{target} tracks (ratio: {ratio:.2f}) {status}"
                )

    # Save dataframes to parquet if requested
    if save_dir and (save_train or save_val or save_test):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving splits to: {save_path}")
        
        if save_train:
            train_path = save_path / "train_df.parquet"
            if is_polars:
                train_df.write_parquet(train_path)
            else:
                train_df.to_parquet(train_path)
            print(f"   ✅ Saved train_df: {train_path}")
            split_info["train_path"] = str(train_path)
        
        if save_val:
            val_path = save_path / "val_df.parquet"
            if is_polars:
                val_df.write_parquet(val_path)
            else:
                val_df.to_parquet(val_path)
            print(f"   ✅ Saved val_df: {val_path}")
            split_info["val_path"] = str(val_path)
        
        if save_test:
            test_path = save_path / "test_df.parquet"
            if is_polars:
                test_df.write_parquet(test_path)
            else:
                test_df.to_parquet(test_path)
            print(f"   ✅ Saved test_df: {test_path}")
            split_info["test_path"] = str(test_path)

    return train_df, val_df, test_df, split_info


def _stratified_track_split(track_info, val_fraction, random_seed=42):
    """Helper function for stratified splitting by condition"""
    np.random.seed(random_seed)

    train_tracks = []
    val_tracks = []

    for condition in track_info["condition"].unique():
        condition_tracks = track_info[track_info["condition"] == condition][
            "unique_id"
        ].tolist()
        np.random.shuffle(condition_tracks)

        n_val = int(len(condition_tracks) * val_fraction)
        val_tracks.extend(condition_tracks[:n_val])
        train_tracks.extend(condition_tracks[n_val:])

    return train_tracks, val_tracks


def move_tracks_to_test_by_window_uids(
    train_df,
    val_df, 
    test_df,
    window_uids_to_move,
    windowed_df=None,
    track_col='unique_id',
):
    """
    Move entire tracks to test set based on specified window_uids.
    
    Given a list of window_uids (e.g., curated ground truth windows), this function
    finds the tracks (unique_ids) that contain those windows and moves ALL data
    from those tracks into the test set. This ensures ground truth windows are
    available for validation while maintaining data integrity (no track splitting).
    
    Parameters
    ----------
    train_df : pl.DataFrame or pd.DataFrame
        Training set instant_df
    val_df : pl.DataFrame or pd.DataFrame
        Validation set instant_df
    test_df : pl.DataFrame or pd.DataFrame
        Test set instant_df
    window_uids_to_move : list of str
        List of window_uid strings to ensure are in test set.
        Format: '{unique_id}_{time_window}_{start_frame}_{end_frame}'
        e.g., 'eekrw_131_7338_R1_1_2849.0_2908.0'
    windowed_df : pl.DataFrame or pd.DataFrame, optional
        The time_windowed_df containing window_uid and unique_id columns.
        If provided, used to lookup unique_ids accurately.
        If None, unique_ids are parsed from window_uid strings.
    track_col : str, default='unique_id'
        Column name for track identifier
        
    Returns
    -------
    tuple
        (train_df_new, val_df_new, test_df_new, move_info)
        
        move_info : dict containing:
            - 'tracks_moved': list of unique_ids moved to test
            - 'from_train': number of tracks moved from train
            - 'from_val': number of tracks moved from val
            - 'already_in_test': number of tracks already in test
            - 'window_uids_found': window_uids that were found
            - 'window_uids_not_found': window_uids that couldn't be matched
            
    Examples
    --------
    >>> curated_windows = ['track1_0_0.0_60.0', 'track2_1_30.0_90.0']
    >>> train_new, val_new, test_new, info = move_tracks_to_test_by_window_uids(
    ...     train_df, val_df, test_df, curated_windows, windowed_df=windowed_df
    ... )
    >>> print(f"Moved {len(info['tracks_moved'])} tracks to test")
    """
    import re
    
    # Check if inputs are Polars or Pandas
    is_polars = hasattr(train_df, 'to_pandas')
    
    # Flatten window_uids list (in case nested dict was passed)
    if isinstance(window_uids_to_move, dict):
        flat_uids = []
        for category, uids in window_uids_to_move.items():
            flat_uids.extend(uids)
        window_uids_to_move = list(set(flat_uids))
    else:
        window_uids_to_move = list(set(window_uids_to_move))
    
    print(f"🔄 Moving tracks to test based on {len(window_uids_to_move)} window_uids...")
    
    # -------------------------------------------------------------------------
    # STEP 1: Find unique_ids for the specified window_uids
    # -------------------------------------------------------------------------
    tracks_to_move = set()
    window_uids_found = []
    window_uids_not_found = []
    
    if windowed_df is not None:
        # Use windowed_df for accurate lookup
        print("   Using windowed_df for unique_id lookup...")
        
        if is_polars or hasattr(windowed_df, 'to_pandas'):
            # Polars
            if hasattr(windowed_df, 'to_pandas'):
                uid_lookup = windowed_df.select(['window_uid', track_col]).unique()
                for wuid in window_uids_to_move:
                    match = uid_lookup.filter(pl.col('window_uid') == wuid)
                    if len(match) > 0:
                        tracks_to_move.add(match[track_col][0])
                        window_uids_found.append(wuid)
                    else:
                        window_uids_not_found.append(wuid)
            else:
                # Pandas windowed_df
                for wuid in window_uids_to_move:
                    match = windowed_df[windowed_df['window_uid'] == wuid]
                    if len(match) > 0:
                        tracks_to_move.add(match[track_col].iloc[0])
                        window_uids_found.append(wuid)
                    else:
                        window_uids_not_found.append(wuid)
    else:
        # Parse unique_id from window_uid string
        # Format: {unique_id}_{time_window}_{start_frame}_{end_frame}
        # The tricky part is that unique_id itself can contain underscores
        # Pattern: everything before the last 3 underscore-separated parts
        print("   Parsing unique_ids from window_uid strings...")
        
        # Pattern to match: _<int>_<float>_<float> at the end
        # This captures time_window, start_frame, end_frame
        pattern = r'^(.+)_(\d+)_([\d.]+)_([\d.]+)$'
        
        for wuid in window_uids_to_move:
            match = re.match(pattern, wuid)
            if match:
                unique_id = match.group(1)
                tracks_to_move.add(unique_id)
                window_uids_found.append(wuid)
            else:
                window_uids_not_found.append(wuid)
                print(f"      ⚠️ Could not parse: {wuid}")
    
    print(f"   Found {len(tracks_to_move)} unique tracks from {len(window_uids_found)} window_uids")
    if window_uids_not_found:
        print(f"   ⚠️ Could not find {len(window_uids_not_found)} window_uids")
    
    # -------------------------------------------------------------------------
    # STEP 2: Identify which split each track is currently in
    # -------------------------------------------------------------------------
    if is_polars:
        train_tracks = set(train_df[track_col].unique().to_list())
        val_tracks = set(val_df[track_col].unique().to_list())
        test_tracks = set(test_df[track_col].unique().to_list())
    else:
        train_tracks = set(train_df[track_col].unique())
        val_tracks = set(val_df[track_col].unique())
        test_tracks = set(test_df[track_col].unique())
    
    tracks_from_train = tracks_to_move & train_tracks
    tracks_from_val = tracks_to_move & val_tracks
    tracks_already_in_test = tracks_to_move & test_tracks
    tracks_not_found = tracks_to_move - (train_tracks | val_tracks | test_tracks)
    
    print(f"\n   Track locations:")
    print(f"      From train: {len(tracks_from_train)}")
    print(f"      From val: {len(tracks_from_val)}")
    print(f"      Already in test: {len(tracks_already_in_test)}")
    if tracks_not_found:
        print(f"      Not found in any split: {len(tracks_not_found)}")
    
    # -------------------------------------------------------------------------
    # STEP 3: Move tracks to test
    # -------------------------------------------------------------------------
    if is_polars:
        # Extract rows to move from train
        train_rows_to_move = train_df.filter(pl.col(track_col).is_in(list(tracks_from_train)))
        train_df_new = train_df.filter(~pl.col(track_col).is_in(list(tracks_from_train)))
        
        # Extract rows to move from val
        val_rows_to_move = val_df.filter(pl.col(track_col).is_in(list(tracks_from_val)))
        val_df_new = val_df.filter(~pl.col(track_col).is_in(list(tracks_from_val)))
        
        # Add to test
        test_df_new = pl.concat([test_df, train_rows_to_move, val_rows_to_move])
    else:
        # Pandas version
        train_rows_to_move = train_df[train_df[track_col].isin(tracks_from_train)]
        train_df_new = train_df[~train_df[track_col].isin(tracks_from_train)]
        
        val_rows_to_move = val_df[val_df[track_col].isin(tracks_from_val)]
        val_df_new = val_df[~val_df[track_col].isin(tracks_from_val)]
        
        test_df_new = pd.concat([test_df, train_rows_to_move, val_rows_to_move], ignore_index=True)
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    n_rows_moved = len(train_rows_to_move) + len(val_rows_to_move)
    
    print(f"\n✅ Move complete!")
    print(f"   Rows moved to test: {n_rows_moved:,}")
    print(f"   New split sizes:")
    print(f"      Train: {len(train_df):,} → {len(train_df_new):,}")
    print(f"      Val: {len(val_df):,} → {len(val_df_new):,}")
    print(f"      Test: {len(test_df):,} → {len(test_df_new):,}")
    
    move_info = {
        'tracks_moved': list(tracks_from_train | tracks_from_val),
        'from_train': len(tracks_from_train),
        'from_val': len(tracks_from_val),
        'already_in_test': len(tracks_already_in_test),
        'not_found': len(tracks_not_found),
        'window_uids_found': window_uids_found,
        'window_uids_not_found': window_uids_not_found,
        'rows_moved': n_rows_moved,
    }
    
    return train_df_new, val_df_new, test_df_new, move_info


def train_motion_transformer(
    instant_df,
    window_size=60,
    overlap=30,
    batch_size=64,
    epochs=10,
    val_split=0.15,
    test_split=0.2,
    split_strategy="hybrid_cell",
    device="auto",
    use_tensorboard=False,
    tensorboard_log_dir=None,
    augmentation_strategy="basic",
    save_model_path=None,
    use_scheduler=False,
    condition_factors=None,
    cells_per_condition=8,
    test_condition_col=None,
):
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
        split_strategy: 'hybrid_cell', 'enhanced_hybrid_cell', 'stratified', 'random', 'cell_balanced'
        device: 'auto', 'cuda', or 'cpu'
        use_tensorboard: Enable TensorBoard logging
        tensorboard_log_dir: Directory for TensorBoard logs
        augmentation_strategy: Augmentation strategy
        save_model_path: Path to save trained model
        use_scheduler: Use learning rate scheduler
        condition_factors: List of column names for condition balancing (e.g., ['cell', 'type', 'mol'])
        test_condition_col: Column name to use for test set selection (if None, uses 'condition')

    Returns:
        trainer: Trained MotionTrainer object
        datasets: Dict with train, val, test datasets
        split_info: Information about the data split

    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🚀 Enhanced Motion Transformer Training")
    print(f"   Device: {device}")
    print(f"   Window: {window_size}f, Overlap: {overlap}f")
    print(
        f"   Splits: {1-(val_split+test_split):.1f} train, {val_split:.1f} val, {test_split:.1f} test"
    )

    # Create smart train/val/test split
    train_df, val_df, test_df, split_info = create_smart_train_val_test_split(
        instant_df,
        val_split=val_split,
        test_split=test_split,
        split_strategy=split_strategy,
        condition_factors=condition_factors,
        cells_per_condition=cells_per_condition,
        test_condition_col=test_condition_col,
    )

    # Create datasets
    print("\n🔄 Creating datasets...")
    train_dataset = PandasTrajectoryDataset(train_df, window_size, overlap)
    val_dataset = (
        PandasTrajectoryDataset(val_df, window_size, overlap)
        if len(val_df) > 0
        else None
    )
    test_dataset = PandasTrajectoryDataset(test_df, window_size, overlap)

    print(f"   Train windows: {len(train_dataset)}")
    print(f"   Val windows: {len(val_dataset) if val_dataset else 0}")
    print(f"   Test windows: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset
        else None
    )
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
        use_scheduler=use_scheduler,
    )

    print("\n🚀 Starting training with validation tracking...")
    trainer.train(epochs=epochs)

    # Save model if path provided
    if save_model_path:
        trainer.save_model(save_model_path)

    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    return trainer, datasets, split_info


def train_multi_scale_transformers(
    df,
    scales=None,
    batch_size=64,
    epochs=10,
    test_split=0.2,
    device="auto",
    n_clusters=5,
    use_tensorboard=False,
    augmentation_strategy="basic",
    save_models=True,
    use_scheduler=False,
    single_scale_mode=False,  # NEW: Add option to use only 60f scale with direct mapping
):
    """
    Train transformers across multiple temporal scales or single scale.

    Parameters
    ----------
    single_scale_mode : bool
        If True, only train 60f scale and use direct window_uid mapping.
        If False, use full multi-scale approach.

    """
    if single_scale_mode:
        print(
            "🎯 SINGLE-SCALE MODE: Training only 60f scale with direct window_uid mapping"
        )
        # Override scales to only use 60f
        scales = [{"window_size": 60, "overlap": 30}]

    # Default scales if none provided
    if scales is None:
        scales = [
            {"window_size": 30, "overlap": 15},
            {"window_size": 60, "overlap": 30},
            {"window_size": 120, "overlap": 60},
            {"window_size": 240, "overlap": 120},
        ]

    from datetime import datetime

    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("MULTI-SCALE TRANSFORMER TRAINING")
    print("=" * 60)

    # Setup TensorBoard logging directories
    tensorboard_base_dir = None
    if use_tensorboard:
        try:
            from . import config

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tensorboard_base_dir = os.path.join(
                config.TENSORBOARD_LOGS, f"multi_scale_{timestamp}"
            )
            os.makedirs(tensorboard_base_dir, exist_ok=True)
            print(f"📊 TensorBoard logs will be saved to: {tensorboard_base_dir}")
        except Exception as e:
            print(f"⚠ TensorBoard setup failed: {e}")
            use_tensorboard = False

    scale_results = {}

    # Train transformer for each scale
    for i, scale_config in enumerate(scales):
        window_size = scale_config["window_size"]
        overlap = scale_config["overlap"]

        print(
            f"\n🔬 SCALE {i+1}: {window_size}-frame windows ({overlap}-frame overlap)"
        )
        print(f"   Step size: {window_size - overlap} frames")

        # Check if any tracks are long enough for this scale
        if isinstance(df, pl.DataFrame):
            max_track_length = df.group_by("unique_id").agg(pl.col("frame").count().alias("track_length"))["track_length"].max()
        else:
            max_track_length = df.groupby("unique_id")["frame"].count().max()
        if max_track_length < window_size:
            print(
                f"   ⚠ Skipping scale: No tracks long enough ({max_track_length} < {window_size})"
            )
            continue

        # Setup scale-specific TensorBoard logging
        scale_tensorboard_dir = None
        if use_tensorboard and tensorboard_base_dir:
            scale_tensorboard_dir = os.path.join(
                tensorboard_base_dir, f"scale_{window_size}f"
            )
            os.makedirs(scale_tensorboard_dir, exist_ok=True)

        # Train transformer for this scale
        try:
            # Setup model save path
            scale_model_path = None
            if save_models and tensorboard_base_dir:
                scale_model_path = os.path.join(
                    tensorboard_base_dir, f"model_{window_size}f.pt"
                )

            trainer, datasets, split_info = train_motion_transformer(
                df,
                window_size=window_size,
                overlap=overlap,
                batch_size=batch_size,
                epochs=epochs,
                val_split=0.15,  # Add validation split
                test_split=test_split,
                split_strategy="enhanced_hybrid_cell",  # Use enhanced smart splitting
                device=device,
                use_tensorboard=use_tensorboard,
                tensorboard_log_dir=scale_tensorboard_dir,
                augmentation_strategy=augmentation_strategy,
                save_model_path=scale_model_path,
                use_scheduler=use_scheduler,
            )

            train_dataset = datasets["train"]
            test_dataset = datasets["test"]

            print(
                f"   ✓ Training completed: {len(train_dataset)} train, {len(test_dataset)} test windows"
            )

        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            continue

        # Extract embeddings and cluster
        print("   📊 Extracting embeddings and clustering...")

        all_dataset = PandasTrajectoryDataset(
            df, window_size=window_size, overlap=overlap
        )
        all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)

        embeddings = trainer.extract_embeddings(all_loader)
        cluster_labels = trainer.cluster_embeddings(embeddings, n_clusters=n_clusters)

        # Get metadata
        condition_labels = all_dataset.get_labels()
        track_info = all_dataset.get_track_info()

        print(f"   ✓ Generated {len(embeddings)} embeddings, {n_clusters} clusters")

        # Create scale-specific windowed dataframe
        step_size = window_size - overlap
        scale_windowed_df = pd.DataFrame(
            {
                "time_window": [
                    info["start_frame"] // step_size for info in track_info
                ],
                "unique_id": [info["unique_id"] for info in track_info],
                "cluster": cluster_labels,
                "condition": condition_labels,
                "start_frame": [info["start_frame"] for info in track_info],
                "end_frame": [info["end_frame"] for info in track_info],
                "window_uid": [
                    info["window_uid"] for info in track_info
                ],  # NEW: Include window_uid for mapping
                "window_size": window_size,
                "scale_id": i,
            }
        )

        # Store results for this scale
        scale_results[f"scale_{window_size}f"] = {
            "trainer": trainer,
            "embeddings": embeddings,
            "cluster_labels": cluster_labels,
            "windowed_df": scale_windowed_df,
            "track_info": track_info,
            "window_size": window_size,
            "overlap": overlap,
            "n_windows": len(embeddings),
        }

        print(f"   ✓ Scale {window_size}f completed: {len(scale_windowed_df)} windows")

    print("\n🔗 INTEGRATING MULTI-SCALE RESULTS...")

    if single_scale_mode:
        # Simple direct mapping for single scale
        print("   🎯 SINGLE-SCALE MODE: Using direct window_uid mapping...")
        scale_name = list(scale_results.keys())[0]  # Should be 'scale_60f'
        scale_info = scale_results[scale_name]

        integrated_instant_df = map_single_scale_direct(
            df.copy(), scale_info["windowed_df"], scale_info["cluster_labels"]
        )

        # For single-scale mode, set cluster_scale column
        integrated_instant_df["cluster_scale"] = integrated_instant_df[
            "final_cluster"
        ].apply(lambda x: "60f" if x != -1 else "none")

        # Create signature columns for single scale
        signature_cols = ["cluster_scale_60f"]
        integrated_instant_df["multi_scale_signature"] = integrated_instant_df[
            "cluster_scale_60f"
        ].astype(str)

    else:
        # 🔥 NEW: Use mixed mapping approach for multi-scale
        integrated_instant_df = map_multiscale_clusters_to_instant(
            instant_df=df,
            scale_results=scale_results,
            track_col="unique_id",
            frame_col="frame",
        )

        # 🔥 NEW: Multi-scale cluster assignment (not just shortest priority)
        print("   📊 Creating multi-scale cluster signatures...")

        # Option 1: Shortest scale priority (backward compatibility)
        scale_names_by_size = sorted(
            scale_results.keys(), key=lambda x: scale_results[x]["window_size"]
        )

        integrated_instant_df["final_cluster"] = np.nan
        integrated_instant_df["cluster_scale"] = "none"

        for i, row in integrated_instant_df.iterrows():
            for scale_name in scale_names_by_size:
                scale_suffix = scale_name.replace("scale_", "")
                cluster_val = row[f"cluster_scale_{scale_suffix}"]
                if cluster_val != -1:  # Changed from pd.notna to != -1
                    integrated_instant_df.at[i, "final_cluster"] = cluster_val
                    integrated_instant_df.at[i, "cluster_scale"] = scale_suffix
                    break

        # Create signature columns for multi-scale
        signature_cols = [
            f'cluster_scale_{scale_name.replace("scale_", "")}'
            for scale_name in scale_results
        ]
        integrated_instant_df["multi_scale_signature"] = integrated_instant_df[
            signature_cols
        ].apply(lambda row: "_".join(row.astype(str)), axis=1)

    # Report final mapping results
    if isinstance(integrated_instant_df, pl.DataFrame):
        frame_assignments = integrated_instant_df.filter(pl.col("final_cluster").is_not_null()).height
    else:
        frame_assignments = (~integrated_instant_df["final_cluster"].isna()).sum()
    print(f"   ✅ Mapped clusters to {frame_assignments:,} trajectory points")

    # Create summary statistics
    scale_coverage = {}
    expected_coverage = {}
    for scale_name in scale_results:
        # Extract scale suffix (e.g., '60f' from 'scale_60f')
        scale_suffix = scale_name.replace("scale_", "")

        # Count actual valid assignments (not -1)
        if isinstance(integrated_instant_df, pl.DataFrame):
            coverage = integrated_instant_df.filter(pl.col(f"cluster_scale_{scale_suffix}") != -1).height
        else:
            coverage = (integrated_instant_df[f"cluster_scale_{scale_suffix}"] != -1).sum()
        scale_coverage[scale_name] = coverage

        # 🔍 DEBUGGING: Calculate expected coverage based on track lengths
        window_size = scale_results[scale_name]["window_size"]
        if isinstance(integrated_instant_df, pl.DataFrame):
            tracks = integrated_instant_df.group_by("unique_id").agg(pl.len().alias("track_length"))
            expected_points = tracks.filter(pl.col("track_length") >= window_size)["track_length"].sum()
        else:
            tracks = integrated_instant_df.groupby("unique_id").size()
            expected_points = tracks[tracks >= window_size].sum()
        expected_coverage[scale_name] = expected_points

    print("\n📊 COVERAGE SUMMARY (Valid Assignments Only):")
    print(f"{'Scale':<10} {'Actual':<10} {'Expected':<10} {'Efficiency':<10}")
    print("-" * 45)
    for scale_name in scale_results:
        actual = scale_coverage[scale_name]
        expected = expected_coverage[scale_name]
        efficiency = (actual / expected * 100) if expected > 0 else 0
        if isinstance(integrated_instant_df, pl.DataFrame):
            total_df_len = integrated_instant_df.height
        else:
            total_df_len = len(integrated_instant_df)
        actual_pct = actual / total_df_len * 100
        expected_pct = expected / total_df_len * 100
        print(
            f"{scale_name:<10} {actual:>6,} ({actual_pct:4.1f}%) {expected:>6,} ({expected_pct:4.1f}%) {efficiency:>6.1f}%"
        )

    if isinstance(integrated_instant_df, pl.DataFrame):
        final_coverage = integrated_instant_df.filter(pl.col("final_cluster").is_not_null()).height
        total_points = integrated_instant_df.height
    else:
        final_coverage = (~integrated_instant_df["final_cluster"].isna()).sum()
        total_points = len(integrated_instant_df)
    final_percentage = final_coverage / total_points * 100
    print(f"\n   Final assignment: {final_coverage:,} points ({final_percentage:.1f}%)")

    # 🔍 DEBUGGING: Show mapping efficiency issues
    print("\n🔍 MAPPING EFFICIENCY ANALYSIS:")
    total_expected = sum(expected_coverage.values())
    total_actual = sum(scale_coverage.values())
    if total_expected > 0:
        mapping_efficiency = total_actual / total_expected * 100
        print(f"   Overall mapping efficiency: {mapping_efficiency:.1f}%")
        if mapping_efficiency < 50:
            print("   ⚠️ Low mapping efficiency suggests coordinate mismatch issues")
            print("   💡 This explains why coverage is similar across scales")

    print("\n📊 WINDOW CREATION vs FRAME MAPPING:")
    for scale_name, scale_info in scale_results.items():
        n_windows = scale_info["n_windows"]
        mapped_points = scale_coverage[scale_name]
        window_size = scale_info["window_size"]
        theoretical_points = n_windows * window_size
        print(
            f"   {scale_name}: {n_windows:,} windows → {theoretical_points:,} theoretical points → {mapped_points:,} actual points"
        )

    # 🔥 NEW: Multi-scale analysis summary
    print("\n📊 MULTI-SCALE BEHAVIORAL ANALYSIS:")
    if isinstance(integrated_instant_df, pl.DataFrame):
        total_frames = integrated_instant_df.height
        
        # Count frames with different levels of scale coverage using Polars
        all_neg_one = pl.all_horizontal([pl.col(col) == -1 for col in signature_cols])
        any_neg_one = pl.any_horizontal([pl.col(col) == -1 for col in signature_cols])
        
        no_coverage = integrated_instant_df.filter(all_neg_one).height
        partial_coverage = integrated_instant_df.filter(any_neg_one & ~all_neg_one).height
        full_coverage = integrated_instant_df.filter(~any_neg_one).height
    else:
        total_frames = len(integrated_instant_df)
        
        # Count frames with different levels of scale coverage using Pandas
        no_coverage = (integrated_instant_df[signature_cols] == -1).all(axis=1).sum()
        partial_coverage = (
            (integrated_instant_df[signature_cols] == -1).any(axis=1)
            & ~(integrated_instant_df[signature_cols] == -1).all(axis=1)
        ).sum()
        full_coverage = (~(integrated_instant_df[signature_cols] == -1).any(axis=1)).sum()

    print(
        f"   🔍 No scale coverage: {no_coverage:,} points ({no_coverage/total_frames*100:.1f}%)"
    )
    print(
        f"   📊 Partial coverage: {partial_coverage:,} points ({partial_coverage/total_frames*100:.1f}%)"
    )
    print(
        f"   🎯 Full multi-scale: {full_coverage:,} points ({full_coverage/total_frames*100:.1f}%)"
    )

    # Sample of multi-scale signatures
    print("\n🧬 Top 5 behavioral signatures:")
    if isinstance(integrated_instant_df, pl.DataFrame):
        valid_signatures = (
            integrated_instant_df
            .filter(pl.col("final_cluster").is_not_null())
            .group_by("multi_scale_signature")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(5)
            .to_pandas()
        )
        for _, row in valid_signatures.iterrows():
            print(f"   {row['multi_scale_signature']}: {row['count']:,} frames")
    else:
        valid_signatures = (
            integrated_instant_df[integrated_instant_df["final_cluster"].notna()][
                "multi_scale_signature"
            ]
            .value_counts()
            .head(5)
        )
        for signature, count in valid_signatures.items():
            print(f"   {signature}: {count:,} frames")

    # Combine all windowed dataframes
    all_windowed_dfs = []
    for scale_name, scale_info in scale_results.items():
        windowed_df = scale_info["windowed_df"].copy()
        windowed_df["scale_name"] = scale_name
        all_windowed_dfs.append(windowed_df)

    combined_windowed_df = (
        pd.concat(all_windowed_dfs, ignore_index=True)
        if all_windowed_dfs
        else pd.DataFrame()
    )

    # Final results package
    results = {
        "integrated_instant_df": integrated_instant_df,
        "combined_windowed_df": combined_windowed_df,
        "scale_results": scale_results,
        "scale_coverage": scale_coverage,
        "scales_config": scales,
        "parameters": {
            "n_clusters": n_clusters,
            "batch_size": batch_size,
            "epochs": epochs,
            "test_split": test_split,
        },
    }

    print("\n✅ MULTI-SCALE ANALYSIS COMPLETED!")
    print(f"   📊 Scales trained: {len(scale_results)}")
    print(
        f"   🎯 Total windows: {sum(s['n_windows'] for s in scale_results.values()):,}"
    )
    print(f"   📍 Frame coverage: {final_percentage:.1f}%")

    return results


def map_single_scale_direct(
    instant_df: pd.DataFrame, windowed_df: pd.DataFrame, cluster_labels: np.ndarray
) -> pd.DataFrame:
    """
    Simple direct mapping using window_uid for single-scale mode.
    This bypasses all the complex multi-scale logic.
    """
    print("   🎯 Using direct window_uid mapping for single scale...")

    # Always initialize the final_cluster column
    result_df = instant_df.copy()
    result_df["final_cluster"] = -1  # Initialize with -1 (unassigned)
    result_df["cluster_scale_60f"] = -1  # Initialize scale-specific column

    # Check for required columns
    if "window_uid" not in instant_df.columns:
        print("   ❌ ERROR: No window_uid column in instant_df")
        print("   💡 Make sure your data was processed with the updated features.py")
        return result_df

    # Create windowed dataframe with cluster assignments
    # The windowed_df from transformer should be a list of track info dicts
    if isinstance(windowed_df, list):
        # Convert list of track info to DataFrame
        windowed_data = []
        for i, window_info in enumerate(windowed_df):
            if i < len(cluster_labels):
                windowed_data.append(
                    {
                        "window_uid": window_info["window_uid"],
                        "unique_id": window_info["unique_id"],
                        "start_frame": window_info["start_frame"],
                        "end_frame": window_info["end_frame"],
                        "time_window": window_info["time_window"],
                        "cluster_scale_60f": cluster_labels[i],
                    }
                )
        windowed_with_clusters = pd.DataFrame(windowed_data)
    else:
        # Assume it's already a DataFrame
        windowed_with_clusters = windowed_df.copy()
        if "cluster_scale_60f" not in windowed_with_clusters.columns:
            windowed_with_clusters["cluster_scale_60f"] = cluster_labels

    if "window_uid" not in windowed_with_clusters.columns:
        print("   ❌ ERROR: No window_uid column in windowed_df")
        print("   💡 The transformer windowed data doesn't have window_uid")
        return result_df

    # Create mapping dictionary from windowed data
    window_cluster_map = dict(
        zip(
            windowed_with_clusters["window_uid"],
            windowed_with_clusters["cluster_scale_60f"],
            strict=False,
        )
    )

    print(f"   📊 Created mapping for {len(window_cluster_map)} windows")

    # Map clusters to instant dataframe using window_uid
    mapped_count = 0
    for idx, row in result_df.iterrows():
        window_uid = row["window_uid"]
        if pd.notna(window_uid) and window_uid in window_cluster_map:
            cluster_id = window_cluster_map[window_uid]
            result_df.at[idx, "cluster_scale_60f"] = cluster_id
            result_df.at[idx, "final_cluster"] = cluster_id
            mapped_count += 1

    print(f"   ✅ Successfully mapped {mapped_count:,} frames using window_uid")
    coverage = (mapped_count / len(result_df)) * 100
    print(f"   📊 Coverage: {coverage:.1f}%")

    return result_df


def map_clusters_to_instant_relative_to_60f(
    instant_df: pd.DataFrame,
    windowed_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    scale_name: str,
) -> pd.DataFrame:
    """
    Map windowed clusters back to instant dataframe using a relative approach:
    - For 60f scale: Use direct window_uid mapping (matches original windowing)
    - For other scales: Map relative to 60f scale based on frame overlap
    - If mapping fails: Assign NaN instead of using fallback methods

    Parameters
    ----------
    instant_df : pd.DataFrame
        The instant trajectory dataframe
    windowed_df : pd.DataFrame
        The windowed dataframe for this specific scale
    cluster_labels : np.ndarray
        Cluster assignments for each window
    scale_name : str
        Name of the scale (e.g., '30f', '60f', '120f', '240f')

    Returns
    -------
    pd.DataFrame
        Updated instant dataframe with cluster assignments

    """
    print(f"   🔧 {scale_name}: Using relative mapping approach...")

    # Add cluster assignments to windowed_df
    windowed_with_clusters = windowed_df.copy()
    windowed_with_clusters[f"cluster_scale_{scale_name}"] = cluster_labels

    # Initialize the cluster column in instant_df
    cluster_col = f"cluster_scale_{scale_name}"
    instant_df[cluster_col] = -1  # Default to -1 (unassigned)

    if scale_name == "scale_60f":
        # Direct window_uid mapping for 60f scale
        print(
            f"   ✅ {scale_name}: Using direct window_uid mapping (matches original windowing)..."
        )

        if "window_uid" not in instant_df.columns:
            print(f"   ⚠️ No window_uid column in instant_df for {scale_name}")
            return instant_df
        if "window_uid" not in windowed_with_clusters.columns:
            print(f"   ⚠️ No window_uid column in windowed_df for {scale_name}")
            return instant_df

        # Create mapping dictionary from windowed data
        window_cluster_map = dict(
            zip(
                windowed_with_clusters["window_uid"],
                windowed_with_clusters[cluster_col],
                strict=False,
            )
        )

        # Map clusters to instant dataframe using window_uid
        instant_df[cluster_col] = (
            instant_df["window_uid"].map(window_cluster_map).fillna(-1).astype(int)
        )

        # Report mapping quality
        mapped_count = (instant_df[cluster_col] != -1).sum()
        total_count = len(instant_df)
        coverage = (mapped_count / total_count) * 100
        print(
            f"   ✅ {scale_name}: Direct window_uid mapping - {mapped_count:,}/{total_count:,} points ({coverage:.1f}% coverage)"
        )

    else:
        # For other scales, map relative to 60f scale based on frame overlap
        print(f"   🔄 {scale_name}: Mapping relative to 60f scale...")

        # Check if 60f clusters exist
        if "cluster_scale_60f" not in instant_df.columns:
            print(
                f"   ⚠️ No 60f clusters available for relative mapping of {scale_name}"
            )
            return instant_df

        # Extract window size from scale name
        try:
            window_size = int(scale_name.replace("f", ""))
        except ValueError:
            print(f"   ⚠️ Could not extract window size from {scale_name}")
            return instant_df

        # Map based on frame overlap with windowed data
        mapped_count = 0
        for _, window_row in windowed_with_clusters.iterrows():
            cluster_id = window_row[cluster_col]

            # Try to find overlapping frames in instant_df
            # Use track and approximate frame range matching
            track_id = window_row["unique_id"]

            # Get the track data from instant_df
            track_mask = instant_df["unique_id"] == track_id
            track_data = instant_df[track_mask].copy()

            if len(track_data) == 0:
                continue

            # For this scale's window, try to find the best matching frames
            # Based on the window center and size
            try:
                # Estimate window center frame from windowed data
                # This is approximate since we don't have exact frame mapping
                track_frames = track_data["frame"].values
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
                    window_end_idx = min(
                        len(track_frames), window_start_idx + window_size
                    )
                    frame_indices = track_data.iloc[
                        window_start_idx:window_end_idx
                    ].index

                # Assign cluster to these frames
                instant_df.loc[frame_indices, cluster_col] = cluster_id
                mapped_count += len(frame_indices)

            except Exception:
                # If mapping fails for this window, skip it
                continue

        # Report mapping quality
        total_count = len(instant_df)
        coverage = (mapped_count / total_count) * 100
        print(
            f"   📊 {scale_name}: Relative mapping - {mapped_count:,}/{total_count:,} points ({coverage:.1f}% coverage)"
        )

    return instant_df


def map_multiscale_clusters_to_instant(
    instant_df: pd.DataFrame,
    scale_results: dict,
    track_col: str = "unique_id",
    frame_col: str = "frame",
) -> pd.DataFrame:
    """
    Map clusters from multiple scales to instant dataframe using relative approach.
    Processes 60f scale first (direct window_uid mapping), then other scales relative to 60f.
    """
    print("   🔧 Using relative mapping approach (60f first, others relative)...")

    result_df = instant_df.copy()

    # Process 60f scale first (if it exists)
    if "scale_60f" in scale_results:
        print("   🔄 Processing scale_60f first (direct window_uid mapping)...")
        scale_info = scale_results["scale_60f"]
        result_df = map_clusters_to_instant_relative_to_60f(
            result_df,
            scale_info["windowed_df"],
            scale_info["cluster_labels"],
            "scale_60f",
        )

    # Then process other scales relative to 60f
    for scale_name, scale_info in scale_results.items():
        if scale_name == "scale_60f":
            continue  # Already processed

        print(f"   🔄 Processing {scale_name}...")
        result_df = map_clusters_to_instant_relative_to_60f(
            result_df,
            scale_info["windowed_df"],
            scale_info["cluster_labels"],
            scale_name,
        )

    return result_df


def map_clusters_to_time_windowed_df(
    time_windowed_df: pd.DataFrame,
    transformer_windows: list,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Step 1: Map transformer clusters to time_windowed_df using window_uid.

    This is the direct mapping step - transformer windows should have the same
    window_uid format as time_windowed_df: unique_id_timewindow_framestart_frameend

    Parameters
    ----------
    time_windowed_df : pd.DataFrame
        Your original time_windowed_df with window_uid column
    transformer_windows : list
        List of window info dicts from transformer dataset.get_track_info()
    cluster_labels : np.ndarray
        Cluster assignments from clustering the transformer embeddings

    Returns
    -------
    pd.DataFrame
        time_windowed_df with cluster column added

    """
    print("🔗 Step 1: Mapping clusters to time_windowed_df using window_uid...")
    print("🔍 DETAILED DIAGNOSTICS:")

    # Create window_uid -> cluster mapping
    window_cluster_map = {}
    for i, window_info in enumerate(transformer_windows):
        if i < len(cluster_labels):
            window_uid = window_info["window_uid"]
            cluster_id = cluster_labels[i]
            window_cluster_map[window_uid] = cluster_id

    print(f"   📊 Created mapping for {len(window_cluster_map)} transformer windows")

    # Diagnostic: Check window_uid formats
    print("\n🔍 WINDOW_UID FORMAT ANALYSIS:")

    # Sample transformer window_uids
    transformer_sample = list(window_cluster_map.keys())[:5]
    print("   📋 Sample transformer window_uids:")
    for uid in transformer_sample:
        print(f"      • {uid}")

    # Sample time_windowed_df window_uids
    windowed_sample = time_windowed_df["window_uid"].head(5).tolist()
    print("   📋 Sample time_windowed_df window_uids:")
    for uid in windowed_sample:
        print(f"      • {uid}")

    # Check for exact matches
    transformer_uids = set(window_cluster_map.keys())
    windowed_uids = set(time_windowed_df["window_uid"].tolist())

    matches = transformer_uids.intersection(windowed_uids)
    transformer_only = transformer_uids - windowed_uids
    windowed_only = windowed_uids - transformer_uids

    print("\n🎯 MATCHING ANALYSIS:")
    print(f"   ✅ Exact matches: {len(matches)}")
    print(f"   🔴 Transformer only: {len(transformer_only)}")
    print(f"   🔵 Time_windowed only: {len(windowed_only)}")

    if len(transformer_only) > 0:
        print("   📋 Sample transformer-only window_uids:")
        for uid in list(transformer_only)[:3]:
            print(f"      • {uid}")

    if len(windowed_only) > 0:
        print("   📋 Sample windowed-only window_uids:")
        for uid in list(windowed_only)[:3]:
            print(f"      • {uid}")

    # Add cluster column to time_windowed_df
    result_df = time_windowed_df.copy()
    result_df["cluster"] = result_df["window_uid"].map(window_cluster_map)

    # Report mapping success
    mapped_count = result_df["cluster"].notna().sum()
    total_count = len(result_df)
    coverage = (mapped_count / total_count) * 100

    print(
        f"\n   ✅ Successfully mapped {mapped_count:,}/{total_count:,} windows ({coverage:.1f}% coverage)"
    )

    return result_df


def map_clusters_from_windowed_to_instant_df(
    instant_df: pd.DataFrame, time_windowed_df_with_clusters: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 2: Map clusters from time_windowed_df to instant_df using existing window_uid.

    This uses the pre-calculated window_uid mapping that already exists between
    time_windowed_df and instant_df (created when features were calculated).

    Parameters
    ----------
    instant_df : pd.DataFrame
        Your original instant_df with window_uid column
    time_windowed_df_with_clusters : pd.DataFrame
        time_windowed_df with cluster column added from step 1

    Returns
    -------
    pd.DataFrame
        instant_df with cluster column added

    """
    print(
        "🔗 Step 2: Mapping clusters from time_windowed_df to instant_df using window_uid..."
    )

    # Create window_uid -> cluster mapping from time_windowed_df
    window_cluster_map = {}
    for _, row in time_windowed_df_with_clusters.iterrows():
        window_uid = row["window_uid"]
        cluster_id = row["cluster"]
        if pd.notna(cluster_id):
            window_cluster_map[window_uid] = cluster_id

    print(f"   📊 Created mapping for {len(window_cluster_map)} windows with clusters")

    # Add cluster column to instant_df using existing window_uid
    result_df = instant_df.copy()
    result_df["cluster"] = result_df["window_uid"].map(window_cluster_map)

    # Report mapping success
    mapped_count = result_df["cluster"].notna().sum()
    total_count = len(result_df)
    coverage = (mapped_count / total_count) * 100

    print(
        f"   ✅ Successfully mapped {mapped_count:,}/{total_count:,} trajectory points ({coverage:.1f}% coverage)"
    )

    return result_df


def create_enhanced_hybrid_split_demo(instant_df, condition_factors=["cell", "type", "mol"]):
    """
    Demo function to show the enhanced hybrid split with clean conditions.
    
    Args:
        instant_df: DataFrame with trajectory data
        condition_factors: List of columns to use for condition balancing
    
    Returns:
        Dictionary with split results and clean condition examples
    """
    print("🚀 Enhanced Hybrid Split Demo")
    print("=" * 50)
    
    # Show original vs clean conditions
    print(f"🔧 Using condition factors: {condition_factors}")
    
    # Create the split
    train_df, val_df, test_df, split_info = create_smart_train_val_test_split(
        instant_df,
        split_strategy="enhanced_hybrid_cell",
        condition_factors=condition_factors,
        test_split=0.2,
        val_split=0.15
    )
    
    print(f"\n✅ Split completed successfully!")
    print(f"   Train: {len(train_df):,} points")
    print(f"   Val: {len(val_df):,} points") 
    print(f"   Test: {len(test_df):,} points")
    
    # Show sample clean conditions
    if "clean_condition" in train_df.columns:
        if hasattr(train_df, 'unique'):  # Polars
            sample_conditions = train_df["clean_condition"].unique()[:10].to_list()
        else:  # Pandas
            sample_conditions = train_df["clean_condition"].unique()[:10].tolist()
        
        print(f"\n🧬 Sample clean conditions:")
        for i, condition in enumerate(sample_conditions, 1):
            print(f"   {i:2d}. {condition}")
    
    return {
        "train_df": train_df,
        "val_df": val_df, 
        "test_df": test_df,
        "split_info": split_info,
        "condition_factors": condition_factors
    }


def cluster_test_set_only_with_mapping(
    trainer,
    datasets: dict,
    time_windowed_df: pd.DataFrame,
    instant_df: pd.DataFrame,
    n_clusters: int = 5,
    cluster_method: str = "kmeans",
) -> dict:
    """
    Complete test-set-only pipeline: Extract test embeddings, cluster, and map to filtered dataframes.

    This is the scientifically rigorous approach that:
    1. Clusters TEST embeddings only
    2. Filters time_windowed_df to TEST cells only
    3. Filters instant_df to TEST cells only
    4. Maps test clusters to test-only dataframes using window_uid

    Parameters
    ----------
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

    Returns
    -------
    dict
        Results with test-only dataframes containing cluster assignments

    """
    print("🧪 TEST-SET-ONLY Transformer Clustering with window_uid Mapping")
    print("=" * 70)

    # Step 1: Extract test embeddings and cluster
    print("📊 Step 1: Extracting and clustering TEST embeddings...")
    from torch.utils.data import DataLoader

    test_dataset = datasets["test"]
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
        track_id = window_info["unique_id"]
        track_condition = window_info["condition"]
        test_cell_filenames.add((track_id, track_condition))

    # Get actual filenames from instant_df for these tracks
    test_track_ids = [info["unique_id"] for info in transformer_windows]
    test_filenames = instant_df[instant_df["unique_id"].isin(test_track_ids)][
        "filename"
    ].unique()

    print(f"   ✅ Test tracks: {len(test_track_ids)} unique tracks")
    print(f"   ✅ Test cells: {len(test_filenames)} unique filenames")
    print(
        f"   📋 Test filenames: {list(test_filenames)[:5]}{'...' if len(test_filenames) > 5 else ''}"
    )

    # Step 3: Filter time_windowed_df to test cells only
    print("\n🔍 Step 3: Filtering time_windowed_df to test cells only...")
    if "filename" in time_windowed_df.columns:
        test_time_windowed_df = time_windowed_df[
            time_windowed_df["filename"].isin(test_filenames)
        ].copy()
    else:
        # Fallback: filter by unique_id
        test_time_windowed_df = time_windowed_df[
            time_windowed_df["unique_id"].isin(test_track_ids)
        ].copy()

    print(f"   📊 Original time_windowed_df: {len(time_windowed_df)} windows")
    print(f"   📊 Test-only time_windowed_df: {len(test_time_windowed_df)} windows")
    print(
        f"   📊 Reduction: {len(time_windowed_df) - len(test_time_windowed_df)} windows filtered out"
    )

    # Step 4: Filter instant_df to test cells only
    print("\n🔍 Step 4: Filtering instant_df to test cells only...")
    test_instant_df = instant_df[instant_df["filename"].isin(test_filenames)].copy()

    print(f"   📊 Original instant_df: {len(instant_df)} trajectory points")
    print(f"   📊 Test-only instant_df: {len(test_instant_df)} trajectory points")
    print(
        f"   📊 Reduction: {len(instant_df) - len(test_instant_df)} points filtered out"
    )

    # Step 5: Map test clusters to test-only time_windowed_df
    print("\n🔗 Step 5: Mapping clusters to test-only time_windowed_df...")

    # Create window_uid -> cluster mapping from transformer results
    window_cluster_map = {}
    for i, window_info in enumerate(transformer_windows):
        window_uid = window_info["window_uid"]
        cluster_id = test_clusters[i]
        window_cluster_map[window_uid] = cluster_id

    print(f"   📊 Created mapping for {len(window_cluster_map)} test windows")

    # Add cluster column to test time_windowed_df
    test_time_windowed_df_with_clusters = test_time_windowed_df.copy()
    test_time_windowed_df_with_clusters["cluster"] = (
        test_time_windowed_df_with_clusters["window_uid"].map(window_cluster_map)
    )

    # Report mapping success for windowed data
    windowed_mapped_count = test_time_windowed_df_with_clusters["cluster"].notna().sum()
    windowed_total_count = len(test_time_windowed_df_with_clusters)
    windowed_coverage = (windowed_mapped_count / windowed_total_count) * 100

    print(
        f"   ✅ Windowed mapping: {windowed_mapped_count:,}/{windowed_total_count:,} windows ({windowed_coverage:.1f}% coverage)"
    )

    # Step 6: Map clusters from test windowed to test instant using window_uid
    print("\n🔗 Step 6: Mapping clusters from test windowed to test instant...")

    # Create window_uid -> cluster mapping from test windowed data
    windowed_cluster_map = {}
    for _, row in test_time_windowed_df_with_clusters.iterrows():
        window_uid = row["window_uid"]
        cluster_id = row["cluster"]
        if pd.notna(cluster_id):
            windowed_cluster_map[window_uid] = cluster_id

    # Add cluster column to test instant_df
    test_instant_df_with_clusters = test_instant_df.copy()
    test_instant_df_with_clusters["cluster"] = test_instant_df_with_clusters[
        "window_uid"
    ].map(windowed_cluster_map)

    # Report mapping success for instant data
    instant_mapped_count = test_instant_df_with_clusters["cluster"].notna().sum()
    instant_total_count = len(test_instant_df_with_clusters)
    instant_coverage = (instant_mapped_count / instant_total_count) * 100

    print(
        f"   ✅ Instant mapping: {instant_mapped_count:,}/{instant_total_count:,} points ({instant_coverage:.1f}% coverage)"
    )

    # Step 7: Calculate clustering quality metrics
    print("\n📊 Step 7: Calculating clustering quality metrics...")

    import collections

    from sklearn.metrics import davies_bouldin_score, silhouette_score

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

    print("\n🎯 TEST-SET-ONLY CLUSTERING RESULTS:")
    print(
        f"   ✅ Test embeddings: {len(test_embeddings)} windows from {len(test_filenames)} cells"
    )
    print(f"   ✅ Windowed coverage: {windowed_coverage:.1f}%")
    print(f"   ✅ Instant coverage: {instant_coverage:.1f}%")
    print(f"   ✅ Mapping success: {'✅ SUCCESS' if mapping_success else '❌ FAILED'}")

    if mapping_success:
        print("   🚀 Ready for visualization and analysis!")
    else:
        print("   ⚠️  Low coverage - check window_uid consistency")

    return {
        "test_time_windowed_df_with_clusters": test_time_windowed_df_with_clusters,
        "test_instant_df_with_clusters": test_instant_df_with_clusters,
        "test_embeddings": test_embeddings,
        "test_clusters": test_clusters,
        "test_filenames": test_filenames,
        "transformer_windows": transformer_windows,
        "cluster_info": {
            "silhouette_score": test_silhouette,
            "davies_bouldin_score": test_davies_bouldin,
            "cluster_balance": cluster_balance,
            "cluster_counts": cluster_counts,
            "windowed_coverage": windowed_coverage,
            "instant_coverage": instant_coverage,
            "mapping_success": mapping_success,
            "n_test_cells": len(test_filenames),
            "n_test_windows": len(test_embeddings),
            "n_test_points": len(test_instant_df_with_clusters),
        },
    }
