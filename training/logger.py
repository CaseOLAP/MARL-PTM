import os
import json
import torch
from datetime import datetime

class TrainingLogger:
    """
    Handles logging of metrics, status messages, and optional model checkpointing during training.
    """

    def __init__(self, log_dir='logs', checkpoint_dir='checkpoints', verbose=True):
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        self.log_file = None
        self.metrics = []

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')

    def log_message(self, message):
        """
        Logs a single message to file and optionally prints to console.
        """
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
        if self.verbose:
            print(message)

    def log_metrics(self, epoch, metrics_dict):
        """
        Logs a dictionary of metrics to file and internal list.
        """
        entry = {'epoch': epoch, **metrics_dict}
        self.metrics.append(entry)
        self.log_message(f"[Epoch {epoch}] Metrics: {json.dumps(metrics_dict, indent=2)}")

    def save_checkpoint(self, model_dict, epoch):
        """
        Saves PyTorch models to disk for recovery or deployment.
        
        Parameters:
            model_dict (dict): {agent_name: model.state_dict()}
            epoch (int): Current training epoch
        """
        for name, model in model_dict.items():
            path = os.path.join(self.checkpoint_dir, f'{name}_epoch{epoch}.pt')
            torch.save(model.state_dict(), path)
            self.log_message(f"Checkpoint saved: {path}")
