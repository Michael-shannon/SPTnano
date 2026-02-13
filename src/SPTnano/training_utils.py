"""
Training utilities for model management and interruption handling
"""

import json
import os
import signal
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import davies_bouldin_score, silhouette_score


class TrainingInterruptHandler:
    """
    Handle training interruptions gracefully with automatic model saving
    """

    def __init__(self, trainer, save_dir: str, save_prefix: str = "checkpoint"):
        """
        Initialize interrupt handler

        Parameters
        ----------
        trainer : MotionTrainer
            The trainer object to monitor
        save_dir : str
            Directory to save checkpoints
        save_prefix : str
            Prefix for checkpoint files

        """
        self.trainer = trainer
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.interrupted = False

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"🛡️ Interrupt handler active. Checkpoints will be saved to: {save_dir}")

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print(f"\n⚠️ Interrupt signal received (signal {signum})")
        print("🔄 Saving model before exit...")
        self.interrupted = True
        self.save_checkpoint("interrupted")
        print("✅ Model saved successfully!")
        print("💡 You can resume training later using the saved checkpoint.")
        exit(0)

    def save_checkpoint(self, suffix: str = None):
        """
        Save training checkpoint

        Parameters
        ----------
        suffix : str, optional
            Additional suffix for checkpoint name

        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if suffix:
            filename = f"{self.save_prefix}_{suffix}_{timestamp}.pt"
        else:
            filename = (
                f"{self.save_prefix}_epoch_{len(self.trainer.losses)}_{timestamp}.pt"
            )

        filepath = os.path.join(self.save_dir, filename)
        self.trainer.save_model(filepath)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "epoch": len(self.trainer.losses),
            "loss_history": self.trainer.losses,
            "augmentation_strategy": self.trainer.augmentation_strategy,
            "device": str(self.trainer.device),
            "interrupted": self.interrupted,
        }

        metadata_path = filepath.replace(".pt", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return filepath

    def auto_save_training(self, epochs: int, save_every: int = 5):
        """
        Train with automatic checkpoint saving

        Parameters
        ----------
        epochs : int
            Number of epochs to train
        save_every : int
            Save checkpoint every N epochs

        """
        print(f"🚀 Starting training with auto-save every {save_every} epochs")

        for epoch_batch in range(0, epochs, save_every):
            remaining_epochs = min(save_every, epochs - epoch_batch)

            print(
                f"\n📊 Training epochs {epoch_batch + 1}-{epoch_batch + remaining_epochs}"
            )
            self.trainer.train(remaining_epochs)

            # Save checkpoint
            checkpoint_path = self.save_checkpoint()
            print(f"💾 Checkpoint saved: {os.path.basename(checkpoint_path)}")

            if self.interrupted:
                break

        print("✅ Training completed!")


class ModelManager:
    """
    Manage model checkpoints and resumption
    """

    @staticmethod
    def list_checkpoints(checkpoint_dir: str) -> dict[str, dict[str, Any]]:
        """
        List available checkpoints with metadata

        Parameters
        ----------
        checkpoint_dir : str
            Directory containing checkpoints

        Returns
        -------
        dict
            Dictionary of checkpoint info

        """
        checkpoints = {}

        if not os.path.exists(checkpoint_dir):
            return checkpoints

        for file in os.listdir(checkpoint_dir):
            if file.endswith(".pt"):
                checkpoint_path = os.path.join(checkpoint_dir, file)
                metadata_path = checkpoint_path.replace(".pt", "_metadata.json")

                # Load metadata if available
                metadata = {}
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                    except:
                        pass

                # Get file stats
                stat = os.stat(checkpoint_path)

                checkpoints[file] = {
                    "path": checkpoint_path,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "metadata": metadata,
                }

        return checkpoints

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
        """
        Find the most recent checkpoint

        Parameters
        ----------
        checkpoint_dir : str
            Directory containing checkpoints

        Returns
        -------
        str or None
            Path to latest checkpoint

        """
        checkpoints = ModelManager.list_checkpoints(checkpoint_dir)

        if not checkpoints:
            return None

        # Sort by modification time
        latest = max(checkpoints.items(), key=lambda x: x[1]["modified"])

        return latest[1]["path"]

    @staticmethod
    def print_checkpoint_summary(checkpoint_dir: str):
        """
        Print summary of available checkpoints
        """
        checkpoints = ModelManager.list_checkpoints(checkpoint_dir)

        if not checkpoints:
            print(f"📁 No checkpoints found in: {checkpoint_dir}")
            return

        print(f"📁 Checkpoints in: {checkpoint_dir}")
        print("=" * 80)

        for name, info in sorted(
            checkpoints.items(), key=lambda x: x[1]["modified"], reverse=True
        ):
            metadata = info["metadata"]

            print(f"📄 {name}")
            print(f"   Size: {info['size_mb']:.1f} MB")
            print(f"   Modified: {info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")

            if metadata:
                print(f"   Epoch: {metadata.get('epoch', 'Unknown')}")
                print(
                    f"   Augmentation: {metadata.get('augmentation_strategy', 'Unknown')}"
                )
                if metadata.get("loss_history"):
                    latest_loss = metadata["loss_history"][-1]
                    print(f"   Latest Loss: {latest_loss:.4f}")
                if metadata.get("interrupted", False):
                    print("   ⚠️ Training was interrupted")
            print()


def create_training_session(
    trainer, session_name: str, base_dir: str = None
) -> TrainingInterruptHandler:
    """
    Create a managed training session with interruption handling

    Parameters
    ----------
    trainer : MotionTrainer
        The trainer object
    session_name : str
        Name for this training session
    base_dir : str, optional
        Base directory for checkpoints (default: uses config)

    Returns
    -------
    TrainingInterruptHandler
        Configured interrupt handler

    """
    if base_dir is None:
        from . import config

        base_dir = os.path.join(config.SAVED_DATA, "training_checkpoints")

    session_dir = os.path.join(base_dir, session_name)

    return TrainingInterruptHandler(
        trainer=trainer, save_dir=session_dir, save_prefix=f"{session_name}_checkpoint"
    )


def resume_training_from_checkpoint(
    checkpoint_path: str, dataloader, device: str = "auto", additional_epochs: int = 10
) -> "MotionTrainer":
    """
    Resume training from a saved checkpoint

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file
    dataloader : DataLoader
        DataLoader for continued training
    device : str
        Device for training
    additional_epochs : int
        Additional epochs to train

    Returns
    -------
    MotionTrainer
        Resumed trainer object

    """
    from .transformer import MotionTrainer, TransformerMotionEncoder

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model and trainer
    model = TransformerMotionEncoder(input_dim=3)

    # Determine augmentation strategy
    augmentation_strategy = checkpoint.get("augmentation_strategy", "basic")

    trainer = MotionTrainer(
        model=model,
        dataloader=dataloader,
        device=device,
        augmentation_strategy=augmentation_strategy,
    )

    # Load checkpoint
    trainer.load_model(checkpoint_path, load_optimizer=True)

    print(f"🔄 Resumed training from: {os.path.basename(checkpoint_path)}")
    print(f"   Previous epochs: {len(trainer.losses)}")
    print(f"   Augmentation: {augmentation_strategy}")

    if additional_epochs > 0:
        print(f"🚀 Continuing training for {additional_epochs} more epochs...")
        trainer.resume_training(additional_epochs)

    return trainer


class Phase1Evaluator:
    """Comprehensive evaluation for Phase 1 baseline establishment"""

    def __init__(self, trainer, datasets, split_info, embeddings, cluster_labels):
        self.trainer = trainer
        self.datasets = datasets
        self.split_info = split_info
        self.embeddings = embeddings
        self.cluster_labels = cluster_labels
        self.metrics = {}

    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        # 1. Clustering quality metrics
        if len(self.embeddings) > 1 and len(np.unique(self.cluster_labels)) > 1:
            self.metrics["silhouette_score"] = silhouette_score(
                self.embeddings, self.cluster_labels
            )
            self.metrics["davies_bouldin_score"] = davies_bouldin_score(
                self.embeddings, self.cluster_labels
            )
        else:
            self.metrics["silhouette_score"] = 0.0
            self.metrics["davies_bouldin_score"] = float("inf")

        # 2. Training dynamics
        self.metrics["final_train_loss"] = (
            self.trainer.train_losses[-1] if self.trainer.train_losses else None
        )
        self.metrics["final_val_loss"] = (
            self.trainer.val_losses[-1] if self.trainer.val_losses else None
        )
        self.metrics["training_epochs"] = len(self.trainer.train_losses)

        # 3. Train-validation gap
        if self.metrics["final_val_loss"] is not None:
            self.metrics["train_val_gap"] = abs(
                self.metrics["final_train_loss"] - self.metrics["final_val_loss"]
            )
        else:
            self.metrics["train_val_gap"] = None

        # 4. Cluster distribution
        cluster_counts = pd.Series(self.cluster_labels).value_counts()
        self.metrics["n_clusters"] = len(cluster_counts)
        self.metrics["cluster_balance"] = (
            cluster_counts.std() / cluster_counts.mean()
            if len(cluster_counts) > 1
            else 0
        )

        # 5. Dataset split quality
        self.metrics["split_strategy"] = self.split_info["strategy"]
        self.metrics["test_complete_cells"] = self.split_info.get(
            "test_complete_cells", False
        )
        self.metrics["n_train_windows"] = len(self.datasets["train"])
        self.metrics["n_val_windows"] = (
            len(self.datasets["val"]) if self.datasets["val"] else 0
        )
        self.metrics["n_test_windows"] = len(self.datasets["test"])

        # 6. Condition balance evaluation
        if "test_condition_balance" in self.split_info:
            balance_info = self.split_info["test_condition_balance"]
            target_info = self.split_info["target_per_condition"]

            balance_scores = []
            for condition in balance_info.keys():
                actual = balance_info[condition]
                target = target_info[condition]
                if target > 0:
                    ratio = actual / target
                    if 0.8 <= ratio <= 1.2:
                        balance_scores.append(1.0)
                    elif 0.5 <= ratio <= 2.0:
                        balance_scores.append(0.5)
                    else:
                        balance_scores.append(0.0)

            self.metrics["condition_balance_score"] = (
                sum(balance_scores) / len(balance_scores) if balance_scores else 0
            )
        else:
            self.metrics["condition_balance_score"] = None

        return self.metrics

    def print_summary(self):
        """Print a comprehensive but clean summary"""
        print("🎯 PHASE 1 BASELINE EVALUATION COMPLETE")
        print("=" * 60)

        # Training Results
        print("\n📈 TRAINING RESULTS:")
        print(f"   Strategy: {self.split_info['strategy']} splitting")
        print(f"   Epochs: {self.metrics['training_epochs']}")
        print(f"   Final Train Loss: {self.metrics['final_train_loss']:.4f}")
        if self.metrics["final_val_loss"] is not None:
            print(f"   Final Val Loss: {self.metrics['final_val_loss']:.4f}")
            print(f"   Train-Val Gap: {self.metrics['train_val_gap']:.4f}")

        # Dataset Quality
        print("\n📊 DATASET SPLITS:")
        print(f"   Train: {self.metrics['n_train_windows']} windows")
        print(f"   Validation: {self.metrics['n_val_windows']} windows")
        print(f"   Test: {self.metrics['n_test_windows']} windows")
        print(
            f"   Test Complete Cells: {'✅' if self.metrics['test_complete_cells'] else '❌'}"
        )

        # Clustering Quality
        print("\n🎯 CLUSTERING QUALITY:")
        print(f"   Clusters: {self.metrics['n_clusters']}")
        print(
            f"   Silhouette Score: {self.metrics['silhouette_score']:.3f} (higher better)"
        )
        print(
            f"   Davies-Bouldin: {self.metrics['davies_bouldin_score']:.3f} (lower better)"
        )
        print(
            f"   Cluster Balance: {self.metrics['cluster_balance']:.3f} (lower better)"
        )

        # Condition Balance
        if self.metrics["condition_balance_score"] is not None:
            score = self.metrics["condition_balance_score"]
            status = (
                "✅ Good"
                if score >= 0.8
                else "⚠️ Moderate"
                if score >= 0.5
                else "❌ Poor"
            )
            print(f"   Condition Balance: {score:.2f} {status}")

        # Overall Quality Assessment
        print("\n🏆 OVERALL ASSESSMENT:")
        quality_indicators = []

        if self.metrics["silhouette_score"] > 0.5:
            quality_indicators.append("✅ Good cluster separation")
        elif self.metrics["silhouette_score"] > 0.3:
            quality_indicators.append("⚠️ Moderate cluster separation")
        else:
            quality_indicators.append("❌ Poor cluster separation")

        if self.metrics["train_val_gap"] is not None:
            if self.metrics["train_val_gap"] < 0.1:
                quality_indicators.append("✅ Good generalization")
            elif self.metrics["train_val_gap"] < 0.3:
                quality_indicators.append("⚠️ Moderate generalization")
            else:
                quality_indicators.append("❌ Overfitting detected")

        if self.metrics["test_complete_cells"]:
            quality_indicators.append("✅ Clean test visualization")

        for indicator in quality_indicators:
            print(f"   {indicator}")

        print("\n✅ BASELINE ESTABLISHED - Ready for Phase 2!")

    def save_results(self, output_dir):
        """Save results to files for future comparison"""
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics
        metrics_file = os.path.join(output_dir, "phase1_metrics.json")

        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = value.item()
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value

        # Add metadata
        serializable_metrics["timestamp"] = datetime.now().isoformat()
        serializable_metrics["phase"] = "baseline"

        with open(metrics_file, "w") as f:
            json.dump(serializable_metrics, f, indent=2)

        print(f"💾 Results saved to: {metrics_file}")
        return metrics_file


def compare_configurations(baseline_metrics, new_metrics, config_name="New Config"):
    """
    Enhanced comparison function for Phase 2 testing.

    Args:
        baseline_metrics: Dict with baseline metrics
        new_metrics: Dict with new configuration metrics
        config_name: Name of new configuration

    Returns:
        bool: True if new config is better

    """
    print(f"\n📊 COMPARING: {config_name} vs Baseline")
    print("-" * 50)

    improvements = []

    # 1. Validation loss comparison
    if "final_val_loss" in new_metrics and new_metrics["final_val_loss"] is not None:
        if (
            "final_val_loss" in baseline_metrics
            and baseline_metrics["final_val_loss"] is not None
        ):
            val_improvement = (
                baseline_metrics["final_val_loss"] - new_metrics["final_val_loss"]
            )
            if val_improvement > 0.01:
                improvements.append(f"✅ Validation loss: {val_improvement:+.4f}")
            else:
                improvements.append(f"➖ Validation loss: {val_improvement:+.4f}")
        else:
            improvements.append("📊 Validation loss: New metric available")

    # 2. Train-val gap comparison
    if "train_val_gap" in new_metrics and new_metrics["train_val_gap"] is not None:
        gap = new_metrics["train_val_gap"]
        if gap < 0.1:
            improvements.append(f"✅ Train-Val gap: {gap:.4f} (Good generalization)")
        elif gap < 0.3:
            improvements.append(f"⚠️ Train-Val gap: {gap:.4f} (Moderate)")
        else:
            improvements.append(f"❌ Train-Val gap: {gap:.4f} (Overfitting risk)")

    # 3. Clustering quality
    if "silhouette_score" in new_metrics and "silhouette_score" in baseline_metrics:
        sil_diff = (
            new_metrics["silhouette_score"] - baseline_metrics["silhouette_score"]
        )
        if sil_diff > 0.05:
            improvements.append(f"✅ Silhouette: {sil_diff:+.3f}")
        else:
            improvements.append(f"➖ Silhouette: {sil_diff:+.3f}")

    # 4. Condition balance
    if (
        "condition_balance_score" in new_metrics
        and new_metrics["condition_balance_score"] is not None
    ):
        score = new_metrics["condition_balance_score"]
        if score >= 0.8:
            improvements.append(f"✅ Condition balance: {score:.2f} (Well balanced)")
        elif score >= 0.5:
            improvements.append(
                f"⚠️ Condition balance: {score:.2f} (Moderately balanced)"
            )
        else:
            improvements.append(f"❌ Condition balance: {score:.2f} (Poorly balanced)")

    # Print results
    for improvement in improvements:
        print(f"   {improvement}")

    # Overall assessment
    good_count = sum(1 for imp in improvements if imp.startswith("✅"))
    total_count = len(improvements)

    if good_count >= total_count * 0.6:
        print(
            f"\n🎯 VERDICT: {config_name} is BETTER ({good_count}/{total_count} improvements)"
        )
        return True
    else:
        print(
            f"\n📋 VERDICT: {config_name} needs improvement ({good_count}/{total_count} improvements)"
        )
        return False


def create_phase1_visualizations(evaluator, save_path=None):
    """Create clean, publication-ready Phase 1 visualizations"""
    fig = plt.figure(figsize=(16, 10))

    # 1. Training Progress
    ax1 = plt.subplot(2, 3, 1)
    epochs = range(1, len(evaluator.trainer.train_losses) + 1)
    plt.plot(
        epochs,
        evaluator.trainer.train_losses,
        "b-",
        linewidth=2,
        label="Training Loss",
        marker="o",
        markersize=3,
    )

    if evaluator.trainer.val_losses:
        plt.plot(
            epochs,
            evaluator.trainer.val_losses,
            "r-",
            linewidth=2,
            label="Validation Loss",
            marker="s",
            markersize=3,
        )

    plt.title("Training Progress", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Dataset Split Visualization
    ax2 = plt.subplot(2, 3, 2)
    split_data = [
        evaluator.metrics["n_train_windows"],
        evaluator.metrics["n_val_windows"],
        evaluator.metrics["n_test_windows"],
    ]
    split_labels = ["Train", "Val", "Test"]
    split_colors = ["blue", "orange", "green"]

    bars = ax2.bar(split_labels, split_data, color=split_colors, alpha=0.7)
    ax2.set_title(
        f'Dataset Split ({evaluator.split_info["strategy"]})', fontweight="bold"
    )
    ax2.set_ylabel("Number of Windows")
    ax2.grid(True, alpha=0.3)

    for bar, value in zip(bars, split_data, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + bar.get_height() * 0.01,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. Cluster Distribution
    ax3 = plt.subplot(2, 3, 3)
    cluster_counts = pd.Series(evaluator.cluster_labels).value_counts().sort_index()
    bars = ax3.bar(
        cluster_counts.index,
        cluster_counts.values,
        alpha=0.7,
        color=sns.color_palette("husl", len(cluster_counts)),
    )
    ax3.set_title("Cluster Distribution", fontweight="bold")
    ax3.set_xlabel("Cluster ID")
    ax3.set_ylabel("Number of Windows")
    ax3.grid(True, alpha=0.3)

    # 4. Quality Metrics Summary
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis("off")

    # Format conditional values first
    val_loss_str = (
        f"{evaluator.metrics['final_val_loss']:.4f}"
        if evaluator.metrics["final_val_loss"] is not None
        else "N/A"
    )
    train_val_gap_str = (
        f"{evaluator.metrics['train_val_gap']:.4f}"
        if evaluator.metrics["train_val_gap"] is not None
        else "N/A"
    )

    metrics_text = f"""
EVALUATION METRICS
{'='*20}
Silhouette: {evaluator.metrics['silhouette_score']:.3f}
Davies-Bouldin: {evaluator.metrics['davies_bouldin_score']:.3f}
Train Loss: {evaluator.metrics['final_train_loss']:.4f}
Val Loss: {val_loss_str}
Train-Val Gap: {train_val_gap_str}
Clusters: {evaluator.metrics['n_clusters']}
Balance: {evaluator.metrics['cluster_balance']:.3f}
"""

    ax4.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    # 5. Embedding Visualization (if UMAP available)
    ax5 = plt.subplot(2, 3, 5)
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(evaluator.embeddings)

        scatter = ax5.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=evaluator.cluster_labels,
            cmap="tab10",
            alpha=0.6,
            s=1,
        )
        ax5.set_title("UMAP Embedding", fontweight="bold")
        ax5.set_xlabel("UMAP 1")
        ax5.set_ylabel("UMAP 2")
        plt.colorbar(scatter, ax=ax5, label="Cluster ID")
    except ImportError:
        ax5.text(
            0.5,
            0.5,
            "UMAP not available\nInstall with:\nmamba install umap-learn",
            ha="center",
            va="center",
            transform=ax5.transAxes,
        )
        ax5.set_title("UMAP Embedding", fontweight="bold")

    # 6. Quality Status
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    # Quality indicators
    quality_items = []
    if evaluator.metrics["silhouette_score"] > 0.5:
        quality_items.append("✅ Good Clustering")
    elif evaluator.metrics["silhouette_score"] > 0.3:
        quality_items.append("⚠️ Moderate Clustering")
    else:
        quality_items.append("❌ Poor Clustering")

    if evaluator.metrics["train_val_gap"] is not None:
        if evaluator.metrics["train_val_gap"] < 0.1:
            quality_items.append("✅ Good Generalization")
        elif evaluator.metrics["train_val_gap"] < 0.3:
            quality_items.append("⚠️ Moderate Generalization")
        else:
            quality_items.append("❌ Overfitting Risk")

    if evaluator.metrics["test_complete_cells"]:
        quality_items.append("✅ Clean Test Set")

    if evaluator.metrics["condition_balance_score"] is not None:
        if evaluator.metrics["condition_balance_score"] >= 0.8:
            quality_items.append("✅ Balanced Conditions")
        else:
            quality_items.append("⚠️ Condition Imbalance")

    quality_text = "QUALITY ASSESSMENT\n" + "=" * 20 + "\n" + "\n".join(quality_items)
    ax6.text(
        0.05,
        0.95,
        quality_text,
        transform=ax6.transAxes,
        fontsize=11,
        verticalalignment="top",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Visualizations saved to: {save_path}")

    plt.show()
    return fig


def get_epoch_range(loss_list, total):
    """
    Returns (start_epoch, epochs_list) for a loss list that may be shorter than total.
    
    This is useful when some loss components (e.g., adjacent_subwindow_losses) 
    were added partway through training.
    
    Args:
        loss_list: List of loss values (may be shorter than total epochs)
        total: Total number of epochs
    
    Returns:
        tuple: (start_epoch, list of epoch numbers)
    """
    if len(loss_list) == 0:
        return total, []
    start = total - len(loss_list) + 1
    return start, list(range(start, total + 1))


def analyze_loss(name, losses, total_epochs):
    """
    Analyze and print statistics for a loss list.
    
    Args:
        name: Name of the loss (for display)
        losses: List of loss values
        total_epochs: Total number of epochs
    """
    print(f"\n{'─' * 40}")
    print(f"📊 {name}")
    print(f"{'─' * 40}")
    
    if not losses:
        print("   ❌ EMPTY LIST")
        return
    
    arr = np.array(losses)
    start_epoch = total_epochs - len(losses) + 1
    
    print(f"   Length: {len(losses)} entries (epochs {start_epoch}-{total_epochs})")
    print(f"   Min:    {arr.min():.6f}")
    print(f"   Max:    {arr.max():.6f}")
    print(f"   Mean:   {arr.mean():.6f}")
    print(f"   Std:    {arr.std():.6f}")
    
    # Check if constant
    if arr.std() < 1e-10:
        print(f"   ⚠️  WARNING: Values appear CONSTANT!")
    
    # Check for NaN/Inf
    if np.any(np.isnan(arr)):
        print(f"   ⚠️  WARNING: Contains NaN values!")
    if np.any(np.isinf(arr)):
        print(f"   ⚠️  WARNING: Contains Inf values!")
    
    # Print first 5 and last 5 values
    print(f"\n   First 5 values (epochs {start_epoch}-{min(start_epoch+4, total_epochs)}):")
    for i, v in enumerate(losses[:5]):
        print(f"      Epoch {start_epoch + i}: {v:.6f}")
    
    if len(losses) > 10:
        print(f"   ...")
    
    print(f"   Last 5 values (epochs {max(total_epochs-4, start_epoch)}-{total_epochs}):")
    for i, v in enumerate(losses[-5:]):
        ep = total_epochs - 4 + i
        print(f"      Epoch {ep}: {v:.6f}")
