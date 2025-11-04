# SPTnano Transformer Development Documentation

## Overview

This document tracks the development of motion transformers for single particle tracking (SPT) analysis in the SPTnano project. The transformer system is designed to learn motion patterns from trajectory data and classify different types of cellular motion behaviors.

## How the Transformer Learns

### Learning Mechanism: Contrastive Self-Supervised Learning

The transformer uses **contrastive learning** to discover motion patterns without requiring labeled data. This is a self-supervised approach where the model learns by comparing different versions of the same trajectory.

#### The Learning Process

**1. Contrastive Loss Function** (`contrastive_loss` in `transformer.py:343-360`)

The model learns by maximizing agreement between differently augmented versions of the same trajectory:

```python
def contrastive_loss(z_i, z_j, temperature=0.5):
    # z_i, z_j are embeddings of two augmented versions of same trajectory
    # Maximize similarity between positives (same trajectory)
    # Minimize similarity between negatives (different trajectories)
    loss = -torch.log(positives / denominator)
    return loss.mean()
```

**2. Training Loop** (`_train_epoch` in `transformer.py:442-465`)

For each training batch:
1. **Create two views**: Apply augmentation to create two versions of each trajectory
   - Original trajectory → Augmentation → Version 1
   - Original trajectory → Augmentation → Version 2
2. **Encode both versions**: Pass through transformer encoder
   - Version 1 → Transformer → Embedding z1
   - Version 2 → Transformer → Embedding z2
3. **Compute loss**: Contrastive loss encourages z1 ≈ z2 (same trajectory should have similar embeddings)
4. **Backpropagate**: Update model weights using Adam optimizer
5. **Repeat**: Process all batches in training set

**What the Model Learns**:
- Motion patterns that are **invariant to augmentation** (noise, measurement uncertainty)
- **Robust representations** that capture essential motion characteristics
- **Discriminative features** that separate different motion behaviors
- **Temporal dynamics** through attention mechanism over 60-frame windows

#### Example Learning Scenario

```
Input: 60-frame trajectory of HTT protein moving in neuron
↓
Augmentation 1: Add measurement noise → z1 = [0.23, -0.45, 0.67, ...]
Augmentation 2: Add different noise  → z2 = [0.25, -0.43, 0.65, ...]
↓
Loss: Encourages z1 and z2 to be similar (same underlying motion)
      Discourages z1 to be similar to embeddings from other trajectories
↓
Result: Model learns that noise is irrelevant, but motion pattern matters
```

### Purpose of the Validation Set

The validation set serves **four critical purposes** in training:

#### 1. Monitor Overfitting (`_validate_epoch` in `transformer.py:467-490`)

```python
def _validate_epoch(self, dataloader):
    self.model.eval()              # Disable dropout, use batch norm in eval mode
    with torch.no_grad():          # No gradient computation - weights not updated
        # Calculate loss on validation data
```

**Purpose**: Detect when model starts memorizing training data instead of learning generalizable patterns

**How it works**:
- Validation loss computed **without updating weights**
- Training loss ↓ + Validation loss ↑ = **Overfitting detected**
- Validation loss ↓ = Model is generalizing well

**Why it matters**: Prevents the transformer from just memorizing specific trajectories in training set

#### 2. Learning Rate Scheduling (`train` method in `transformer.py:505-508`)

```python
if self.use_scheduler and self.scheduler:
    loss_for_scheduler = val_loss if val_loss is not None else train_loss
    self.scheduler.step(loss_for_scheduler)
```

**Purpose**: Automatically adjust learning rate based on validation performance

**How it works**:
- `ReduceLROnPlateau` monitors validation loss
- If loss plateaus for 3 epochs → reduce learning rate by 50%
- Smaller learning rate helps fine-tune without overshooting

**Why it matters**: Helps model escape local minima and converge to better solutions

#### 3. Model Selection

**Purpose**: Choose the best model checkpoint for deployment

**How it works**:
- Save model when validation loss reaches new minimum
- Final model = epoch with best validation performance (not just lowest training loss)

**Why it matters**: Training loss can be misleadingly low if model is overfit

#### 4. Unbiased Performance Estimation

**Purpose**: Get honest estimate of real-world performance

**How it works**:
- Validation data **never used for gradient updates**
- Completely independent assessment of model quality
- Validation ≠ Test (test set held separate for final evaluation)

**Why it matters**: Training metrics are biased; validation gives realistic performance estimate

### Data Split Strategy

From `config.py:62-69`:

```python
"split_params": {
    "val_split": 0.1,               # 10% of data for validation
    "test_split": 0.2,              # 20% for final testing
    "split_strategy": "fixed_cells", # Keep complete cells together
    "cells_per_condition": 6,       # Balanced test set
}
```

**Smart Splitting Features**:
- **Cell-level splitting**: Entire cells stay together (no data leakage)
- **Condition balancing**: Equal representation across experimental conditions
- **Track-level validation**: Train/val split within cells for efficiency
- **Test set isolation**: Test cells completely held out until final evaluation

**The Three Sets**:
1. **Training Set (70%)**: Used to update model weights via backpropagation
2. **Validation Set (10%)**: Monitor overfitting, tune hyperparameters, no weight updates
3. **Test Set (20%)**: Final evaluation only, never seen during training

**Data Flow During Training**:
```
Epoch 1:
  Training set   → Forward → Loss → Backward → Update weights → Train Loss: 0.45
  Validation set → Forward → Loss → (no backward) → Val Loss: 0.52
  
Epoch 2:
  Training set   → Forward → Loss → Backward → Update weights → Train Loss: 0.38
  Validation set → Forward → Loss → (no backward) → Val Loss: 0.41
  ✓ Validation improving → Continue training
  
Epoch N:
  Train Loss: 0.15 ↓
  Val Loss: 0.35 ↑
  ⚠ Overfitting detected → Stop training or reduce learning rate
```

### Key Takeaway

The validation set ensures your transformer learns **generalizable motion patterns** that work on new, unseen trajectories — not just memorizes your training data. This is essential for discovering biologically meaningful motion behaviors that apply across your experimental conditions.

## Current Implementation Status

### ✅ Completed Components

#### 1. Core Transformer Architecture (`src/SPTnano/transformer.py`)

**TransformerMotionEncoder**
- **Architecture**: Encoder-only transformer with CLS token
- **Input**: Motion sequences `[batch_size, sequence_length, 3]` where features are `[dx, dy, direction]`
- **Parameters**:
  - `input_dim=3`: Motion features (dx, dy, direction)
  - `embed_dim=64`: Embedding dimension
  - `num_heads=4`: Multi-head attention heads
  - `ff_dim=128`: Feed-forward network dimension
  - `num_layers=2`: Number of transformer layers
- **Output**: Fixed-size embeddings `[batch_size, embed_dim]`

**Key Features**:
- CLS token for sequence-level representation
- Layer normalization for stable training
- Configurable architecture parameters

#### 2. Data Processing Pipeline

**PandasTrajectoryDataset**
- Converts trajectory DataFrames to time windows
- **DataFrame Support**: Compatible with both Pandas and Polars DataFrames
- **Window Parameters**:
  - `window_size=60`: Default 60-frame windows
  - `overlap=30`: 50% overlap between windows
  - `min_track_length=60`: Minimum track length requirement
- **Features Extracted**: `[dx, dy, direction_rad]`
- **Metadata Tracking**: Preserves `window_uid` for mapping back to original data

**Smart Data Splitting**
- **Hybrid Cell Strategy**: Complete cells for test set, mixed tracks for train/val
- **Condition-Aware Balancing**: Ensures representative test sets across experimental conditions
- **Strategies Available**: `hybrid_cell`, `stratified`, `random`, `cell_balanced`

#### 3. Training Infrastructure

**MotionTrainer Class**
- **Loss Function**: Contrastive learning with temperature scaling
- **Augmentation**: Configurable augmentation strategies
- **Validation Tracking**: Separate train/validation loss monitoring
- **Learning Rate Scheduling**: ReduceLROnPlateau scheduler
- **TensorBoard Integration**: Comprehensive logging and visualization
- **Model Persistence**: Save/load functionality with training state

**Training Features**:
- Automatic device detection (CUDA/CPU)
- Validation loss tracking
- Learning rate scheduling
- Model checkpointing
- Resume training capability

#### 4. Multi-Scale Analysis

**Multi-Scale Training Pipeline**
- **Scales Supported**: 30f, 60f, 120f, 240f windows
- **Single-Scale Mode**: Option for 60f-only analysis with direct mapping
- **Cluster Mapping**: Maps clusters from windowed data back to instant trajectory points
- **Coverage Analysis**: Detailed reporting of mapping efficiency

**Mapping Strategies**:
- **Direct Window_UID Mapping**: For 60f scale (matches original windowing)
- **Relative Mapping**: For other scales based on frame overlap
- **Multi-Scale Signatures**: Combined behavioral signatures across scales

#### 5. Clustering and Analysis

**Embedding Extraction**
- Extract learned representations from trained models
- Support for both training and test data
- Metadata preservation for mapping

**Clustering Methods**
- **HDBSCAN** (primary method): Automatic cluster discovery with noise detection
  - Auto-discovers optimal number of clusters
  - Identifies outliers as noise points (label -1)
  - Provides cluster stability metrics (persistence scores)
  - Quality metrics: Silhouette score, Davies-Bouldin score, Calinski-Harabasz score
  - Tunable parameters: `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`
- **K-means** (legacy): Fixed number of clusters
  - Configurable cluster numbers
  - Standard quality metrics

**Test-Set-Only Analysis**
- Scientifically rigorous approach using only test data for clustering
- Filters dataframes to test cells only
- Maps clusters using window_uid system

**Interpretability Tools** (NEW)
- **Frame Importance Analysis**: Methods to understand which time points matter most
  - `extract_attention_weights(x, method='gradient')`: Gradient-based importance scores
  - `extract_attention_weights(x, method='leave_one_out')`: Leave-one-out analysis
- **Purpose**: Understand what temporal patterns the model learned
- **Use Cases**: 
  - Identify critical motion events (speed changes, direction changes)
  - Validate model focuses on biologically relevant features
  - Debug unexpected cluster assignments

### 📁 Saved Models

Current saved models in `notebooks/`:
- `best_model.pth`: Best performing model checkpoint
- `htt_motion_transformer.pt`: HTT-specific motion transformer
- `motion_encoder.pt`: General motion encoder model

### 🔧 Key Functions and APIs

#### Training Functions
```python
# Single-scale training
trainer, datasets, split_info = train_motion_transformer(
    instant_df,
    window_size=60,
    overlap=30,
    epochs=10,
    split_strategy="hybrid_cell"
)

# Multi-scale training
results = train_multi_scale_transformers(
    df,
    scales=[
        {"window_size": 30, "overlap": 15},
        {"window_size": 60, "overlap": 30},
        {"window_size": 120, "overlap": 60}
    ],
    epochs=10,
    single_scale_mode=False
)
```

#### Analysis Functions
```python
# Test-set-only clustering with HDBSCAN
import hdbscan
from sklearn.metrics import silhouette_score

# Extract embeddings
embeddings = trainer.extract_embeddings(test_dataloader)

# HDBSCAN clustering (automatic cluster discovery)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100,
    min_samples=50,
    metric='euclidean'
)
cluster_labels = clusterer.fit_predict(embeddings)

# Quality metrics
valid_mask = cluster_labels >= 0
silhouette = silhouette_score(
    embeddings[valid_mask], 
    cluster_labels[valid_mask]
)

# Legacy K-means clustering
clusters = trainer.cluster_embeddings(embeddings, n_clusters=5)
```

#### Interpretability Functions (NEW)
```python
# Extract frame importance scores
sample_batch = next(iter(test_loader))

# Method 1: Gradient-based (fast)
importance = trainer.model.extract_attention_weights(
    sample_batch['features'], 
    method='gradient'
)

# Method 2: Leave-one-out (thorough)
importance = trainer.model.extract_attention_weights(
    sample_batch['features'], 
    method='leave_one_out'
)

# Analyze results
print(f"Most important frame: {importance.mean(0).argmax()}")
print(f"Temporal pattern: {importance.mean(0)}")
```

## Current Challenges and Development Areas

### 🚧 Known Issues

1. **Mapping Efficiency**: Some scales show lower than expected mapping efficiency
2. **Window_UID Consistency**: Need to ensure consistent window_uid generation across pipeline
3. **Multi-Scale Integration**: Complex mapping between different temporal scales

### ⚠️ **CRITICAL: Window_UID Format Mismatch (RESOLVED)**

**Issue**: When loading saved test splits or using inference notebooks, window_uid format can mismatch between transformer metadata and original dataframes.

**Root Cause**: 
- `PandasTrajectoryDataset.get_track_info()` returns window_uid in format: `unique_id_timewindow_framestart_frameend`
- Original instant_df may have window_uid in format: `timewindow_framestart_frameend` (missing unique_id prefix)

**Symptoms**:
- Cluster mapping joins fail silently (0% coverage)
- Window_uid comparison shows no matches between cluster_mapping_df and test_df
- Example mismatch:
  - cluster_mapping: `nvh1h3_0_123_456`
  - test_df: `0_123_456`

**Solution**:
Add unique_id prefix to test_df window_uid before joining:

```python
# For Polars DataFrame
test_df = test_df.with_columns(
    (pl.col('unique_id').cast(pl.Utf8) + '_' + pl.col('window_uid')).alias('window_uid')
)

# For Pandas DataFrame
test_df['window_uid'] = test_df['unique_id'].astype(str) + '_' + test_df['window_uid']
```

**Prevention**:
Always verify window_uid format before mapping:
```python
# Check format
print("Cluster mapping sample:", cluster_mapping_df['window_uid'][0])
print("Test_df sample:", test_df['window_uid'][0])

# Should both show: unique_id_timewindow_framestart_frameend
```

**Status**: Fixed in inference notebooks (October 2025)

### 🎯 Active Development Areas

1. **Architecture Optimization**: Experimenting with different transformer configurations
2. **Augmentation Strategies**: Developing motion-specific data augmentation
3. **Multi-Scale Fusion**: Better integration of information across temporal scales
4. **Evaluation Metrics**: Developing motion-specific evaluation criteria

## Usage Examples

### Basic Training Workflow

```python
import SPTnano as spt

# Load data
instant_df = spt.load_instant_df()

# Train transformer
trainer, datasets, split_info = spt.transformer.train_motion_transformer(
    instant_df,
    window_size=60,
    epochs=20,
    use_tensorboard=True,
    save_model_path="models/motion_transformer.pt"
)

# Extract embeddings and cluster
embeddings = trainer.extract_embeddings()
clusters = trainer.cluster_embeddings(embeddings, n_clusters=5)
```

### Multi-Scale Analysis

```python
# Multi-scale training
results = spt.transformer.train_multi_scale_transformers(
    instant_df,
    epochs=15,
    use_tensorboard=True,
    single_scale_mode=False
)

# Access results
integrated_df = results["integrated_instant_df"]
scale_results = results["scale_results"]
```

## Technical Architecture

### Data Flow
1. **Trajectory Data** → **Time Windows** → **Motion Features** → **Transformer** → **Embeddings** → **Clusters**
2. **Cluster Mapping**: Clusters → Windowed DataFrame → Instant DataFrame (via window_uid)

### Key Design Decisions
- **Contrastive Learning**: Self-supervised approach for learning motion representations
- **CLS Token**: Sequence-level representation for classification
- **Window_UID System**: Enables precise mapping between different data granularities
- **Hybrid Cell Strategy**: Balances training robustness with clean test visualization

## Development Roadmap

### Immediate Priorities
1. **Model Evaluation**: Comprehensive evaluation of existing saved models
2. **Architecture Experiments**: Test different transformer configurations
3. **Augmentation Development**: Motion-specific augmentation strategies
4. **Documentation**: Complete API documentation and examples

### Future Enhancements
1. **Attention Visualization**: Understand what the model learns
2. **Multi-Modal Integration**: Incorporate additional trajectory features
3. **Real-Time Analysis**: Streaming analysis capabilities
4. **Biological Validation**: Correlation with known biological processes

### Recommended Augmentation Improvements

Current: `measurement_noise` (localization uncertainty)

**High Priority Additions** (for next training iteration):

1. **Rotational Invariance**
   - Rotate trajectories by random angles
   - Biological rationale: Cells have random orientations, no preferred direction
   - Implementation: `dx' = dx*cos(θ) - dy*sin(θ), dy' = dx*sin(θ) + dy*cos(θ)`

2. **Mirror Symmetry (Reflection)**
   - Flip X/Y axes randomly
   - Biological rationale: Most SPT motion is achiral (left/right symmetric)
   - Implementation: `dx → -dx` and/or `dy → -dy`

3. **Frame Dropout (Blinking)**
   - Randomly remove 10-20% of frames with interpolation
   - Biological rationale: Fluorophore blinking, out-of-focus events, missed detections
   - Implementation: Drop random frames, interpolate missing positions

4. **Speed Scaling**
   - Scale all displacements by random factor (0.7-1.3x)
   - Biological rationale: Temperature/ATP variation, same pattern at different speeds
   - Implementation: `dx' = α*dx, dy' = α*dy` (preserves directionality)

**Benefit**: These augmentations model real experimental and biological variability specific to HTT protein SPT, improving robustness and biological interpretability.

## Development Notes

### Recent Work (October 2024)
- Implemented multi-scale transformer training pipeline
- Added comprehensive data splitting strategies
- Developed window_uid mapping system for precise cluster assignment
- Created test-set-only analysis pipeline for scientific rigor
- Added TensorBoard integration and model persistence
- **NEW**: Added Polars DataFrame compatibility for large-scale data processing

### Latest Updates (October 16, 2025)

#### Window_UID Format Fix (CRITICAL)
- **Discovered**: Window_uid mismatch between transformer metadata and original dataframes
- **Impact**: Prevented cluster mapping from working (0% join coverage)
- **Solution**: Add unique_id prefix to test_df window_uid before joining
- **Status**: Fixed and documented with prevention checks
- **Location**: See "Known Issues" section for full details and code examples

### Previous Updates (October 15, 2025)

#### HDBSCAN Clustering Implementation
- **Added HDBSCAN**: Replaced K-means as primary clustering method for better cluster discovery
- **Automatic Optimization**: No need to specify number of clusters upfront
- **Noise Detection**: Identifies outliers that don't fit any pattern
- **Quality Metrics**: Comprehensive evaluation with silhouette, Davies-Bouldin, Calinski-Harabasz scores
- **Cluster Stability**: Persistence scores show how robust each cluster is
- **Parameter Tuning**: Easy adjustment via `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`

#### Model Interpretability Enhancements
- **New Methods Added to TransformerMotionEncoder**:
  - `extract_attention_weights(x, method='gradient')`: Fast gradient-based frame importance
  - `extract_attention_weights(x, method='leave_one_out')`: Thorough ablation analysis
  - `forward(x, return_attention=False)`: Extended signature for future attention extraction
- **Private Methods**:
  - `_gradient_based_importance(x)`: Computes sensitivity of embeddings to each frame
  - `_leave_one_out_importance(x)`: Measures impact of removing individual frames
- **Use Case**: Understand which parts of 60-frame windows are most discriminative
- **Interpretation**: High importance = frame is critical for learned representation

#### Benefits
- **HDBSCAN**: More robust clustering with automatic parameter selection
- **Interpretability**: Debug and validate what the model learns
- **Scientific Rigor**: Quality metrics for publication-ready analysis

### Previous Updates (October 10, 2025)

#### Fixed Cells Strategy Refinement
- **Issue Resolved**: Fixed `UnboundLocalError` for `test_track_ids` in fixed_cells strategy
- **Root Cause**: The code was creating `test_df` from selected cells but wasn't extracting `test_track_ids`
- **Solution**: Added proper extraction of track IDs from test set and complete `split_info` dictionary
- **Configuration Update**: Changed default from 8 to 6 cells per condition for better data availability

#### Flexible Test Condition Selection
- **New Parameter**: Added `test_condition_col` to `create_smart_train_val_test_split()` and `train_motion_transformer()`
- **Purpose**: Allow different columns for test set selection vs class balancing
- **Use Case**: Use simple columns like 'mol' for test selection while using complex `class_balance_label` for train/val balancing
- **Example**: Select 6 cells per molecule type (HTT, kinesin, dynein, myosin) instead of per complex condition

#### Notebook Configuration System
- **Two Modes Added**:
  - `FULL_DATASET`: Uses all available data with full training parameters (15 epochs, batch_size=64)
  - `QUICK_TEST`: Filters to subset of data for rapid testing (5 epochs, batch_size=32, 2 molecules only)
- **Smart Filtering**: Automatic filtering by molecule types, max cells per condition, and max total conditions
- **Benefits**: Easy switching between full experiments and quick development iterations
- **Usage**: Simply change `CONFIG_MODE = "QUICK_TEST"` at top of notebook

### Testing Status
- Unit tests exist for core transformer components (`tests/test_transformer.py`)
- Integration tests needed for full pipeline
- Performance benchmarks needed

## Resources and References

### Key Files
- `src/SPTnano/transformer.py`: Main implementation
- `notebooks/TRANSFORMER_10_10.ipynb`: Current development notebook
- `tests/test_transformer.py`: Unit tests
- `src/SPTnano/config.py`: Configuration parameters

### Dependencies
- PyTorch: Deep learning framework
- scikit-learn: Clustering and metrics
- pandas/numpy: Data manipulation
- matplotlib/seaborn: Visualization
- tensorboard: Training monitoring

## Fixed Cells Strategy (NEW - October 2024)

### Overview
The `fixed_cells` strategy provides rigorous experimental design for paper analysis by selecting a fixed number of complete cells per condition for the test set.

### Key Features
- **Test Set**: Exactly N cells per condition (default: 8 cells)
- **Cell Selection**: Uses original `condition` column for balanced selection
- **Train/Val Split**: Remaining data split by tracks (default: 90% train, 10% val)
- **Visualization Ready**: Complete cells ensure clean visualization
- **Statistical Power**: Sufficient test data per condition for analysis

### Configuration
```python
split_params = {
    "split_strategy": "fixed_cells",
    "cells_per_condition": 6,     # Number of cells per condition in test set
    "val_split": 0.1,             # Fraction of remaining data for validation
    "condition_factors": ["mol"], # For creating class balance labels
    "test_condition_col": "mol"   # Column to use for test set selection (optional)
}
```

### Benefits
- Guarantees equal representation of conditions in test set
- Provides sufficient data for statistical analysis (6 cells × conditions)
- Maintains complete cells for visualization
- Flexible condition selection: use simple columns (e.g., 'mol') for test set, complex labels for balancing
- Uses `test_condition_col` for test cell selection, `class_balance_label` for train/val balancing

---

*Last Updated: October 15, 2025*
*Status: Active Development - HDBSCAN Clustering & Model Interpretability Added*
