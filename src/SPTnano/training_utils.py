"""
Training utilities for model management and interruption handling
"""

import os
import signal
import torch
import time
from datetime import datetime
from typing import Optional, Dict, Any
import json

class TrainingInterruptHandler:
    """
    Handle training interruptions gracefully with automatic model saving
    """
    
    def __init__(self, trainer, save_dir: str, save_prefix: str = "checkpoint"):
        """
        Initialize interrupt handler
        
        Parameters:
        -----------
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
        
        Parameters:
        -----------
        suffix : str, optional
            Additional suffix for checkpoint name
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if suffix:
            filename = f"{self.save_prefix}_{suffix}_{timestamp}.pt"
        else:
            filename = f"{self.save_prefix}_epoch_{len(self.trainer.losses)}_{timestamp}.pt"
        
        filepath = os.path.join(self.save_dir, filename)
        self.trainer.save_model(filepath)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'epoch': len(self.trainer.losses),
            'loss_history': self.trainer.losses,
            'augmentation_strategy': self.trainer.augmentation_strategy,
            'device': str(self.trainer.device),
            'interrupted': self.interrupted
        }
        
        metadata_path = filepath.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath
    
    def auto_save_training(self, epochs: int, save_every: int = 5):
        """
        Train with automatic checkpoint saving
        
        Parameters:
        -----------
        epochs : int
            Number of epochs to train
        save_every : int
            Save checkpoint every N epochs
        """
        print(f"🚀 Starting training with auto-save every {save_every} epochs")
        
        for epoch_batch in range(0, epochs, save_every):
            remaining_epochs = min(save_every, epochs - epoch_batch)
            
            print(f"\n📊 Training epochs {epoch_batch + 1}-{epoch_batch + remaining_epochs}")
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
    def list_checkpoints(checkpoint_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        List available checkpoints with metadata
        
        Parameters:
        -----------
        checkpoint_dir : str
            Directory containing checkpoints
            
        Returns:
        --------
        dict
            Dictionary of checkpoint info
        """
        checkpoints = {}
        
        if not os.path.exists(checkpoint_dir):
            return checkpoints
        
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pt'):
                checkpoint_path = os.path.join(checkpoint_dir, file)
                metadata_path = checkpoint_path.replace('.pt', '_metadata.json')
                
                # Load metadata if available
                metadata = {}
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                # Get file stats
                stat = os.stat(checkpoint_path)
                
                checkpoints[file] = {
                    'path': checkpoint_path,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'metadata': metadata
                }
        
        return checkpoints
    
    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """
        Find the most recent checkpoint
        
        Parameters:
        -----------
        checkpoint_dir : str
            Directory containing checkpoints
            
        Returns:
        --------
        str or None
            Path to latest checkpoint
        """
        checkpoints = ModelManager.list_checkpoints(checkpoint_dir)
        
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest = max(checkpoints.items(), 
                    key=lambda x: x[1]['modified'])
        
        return latest[1]['path']
    
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
        
        for name, info in sorted(checkpoints.items(), 
                               key=lambda x: x[1]['modified'], 
                               reverse=True):
            metadata = info['metadata']
            
            print(f"📄 {name}")
            print(f"   Size: {info['size_mb']:.1f} MB")
            print(f"   Modified: {info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            if metadata:
                print(f"   Epoch: {metadata.get('epoch', 'Unknown')}")
                print(f"   Augmentation: {metadata.get('augmentation_strategy', 'Unknown')}")
                if 'loss_history' in metadata and metadata['loss_history']:
                    latest_loss = metadata['loss_history'][-1]
                    print(f"   Latest Loss: {latest_loss:.4f}")
                if metadata.get('interrupted', False):
                    print("   ⚠️ Training was interrupted")
            print()

def create_training_session(trainer, 
                          session_name: str,
                          base_dir: str = None) -> TrainingInterruptHandler:
    """
    Create a managed training session with interruption handling
    
    Parameters:
    -----------
    trainer : MotionTrainer
        The trainer object
    session_name : str
        Name for this training session
    base_dir : str, optional
        Base directory for checkpoints (default: uses config)
        
    Returns:
    --------
    TrainingInterruptHandler
        Configured interrupt handler
    """
    if base_dir is None:
        # Import config from parent directory
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        import config
        base_dir = os.path.join(config.SAVED_DATA, 'training_checkpoints')
    
    session_dir = os.path.join(base_dir, session_name)
    
    return TrainingInterruptHandler(
        trainer=trainer,
        save_dir=session_dir,
        save_prefix=f"{session_name}_checkpoint"
    )

def resume_training_from_checkpoint(checkpoint_path: str,
                                  dataloader,
                                  device: str = 'auto',
                                  additional_epochs: int = 10) -> 'MotionTrainer':
    """
    Resume training from a saved checkpoint
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to checkpoint file
    dataloader : DataLoader
        DataLoader for continued training
    device : str
        Device for training
    additional_epochs : int
        Additional epochs to train
        
    Returns:
    --------
    MotionTrainer
        Resumed trainer object
    """
    from .transformer import TransformerMotionEncoder, MotionTrainer
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model and trainer
    model = TransformerMotionEncoder(input_dim=3)
    
    # Determine augmentation strategy
    augmentation_strategy = checkpoint.get('augmentation_strategy', 'basic')
    
    trainer = MotionTrainer(
        model=model,
        dataloader=dataloader,
        device=device,
        augmentation_strategy=augmentation_strategy
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