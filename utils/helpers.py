import json
import logging
import time
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import os
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols  # List of columns to apply cyclical encoding to
        self.max_values = {}  # Store the max values for each column

    def fit(self, X, y=None):
        """
        Learn the max value for each column during training.
        """
        for col in self.cols:
            max_val = X[col].max()
            self.max_values[col] = max_val
        return self

    def transform(self, X):
        """
        Apply cyclical encoding to the columns using learned max values.
        """
        X_copy = X.copy()
        for col in self.cols:
            # Default to 1 if max_val not found
            max_val = self.max_values.get(col, 1)
            X_copy[col + '_sin'] = np.sin(2 *
                                          np.pi * X_copy[col] / max_val).round(8)
            X_copy[col + '_cos'] = np.cos(2 *
                                          np.pi * X_copy[col] / max_val).round(8)
        return X_copy


class Trainer:
    def __init__(self, model, criterion, optimizer, device, log_dir=None, checkpoint_dir=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.plotting_dict = {
            "train_loss": [],
            "val_loss": [],
        }
        self.early_stop_counter = 0
        self.early_stop_patience = 5
        self.best_val_loss = float('inf')
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory if it doesn't exist
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # TensorBoard setup
        if log_dir is None:
            current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
            log_dir = os.path.join('runs', current_time)

        self.writer = SummaryWriter(log_dir)

    def _run_epoch(self, epoch, epochs, train_dataloader):
        """Run a single training epoch"""
        self.model.train()
        total_loss = 0

        for i, (inputs, targets, category) in enumerate(train_dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            category = category.to(self.device)

            outputs = self.model(inputs, category)
            loss = self.criterion(outputs, targets.float())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

            if i % 10 == 0:
                step = epoch * len(train_dataloader) + i
                self.writer.add_scalar('Batch/Loss', loss.item(), step)

        # Calculate average training loss for the epoch
        train_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}")

        return train_loss

    def _handle_validation(self, epoch, val_dataloader, train_loss):
        """Handle validation, early stopping, and checkpointing"""
        val_loss = self.evaluate(val_dataloader, "Validation")

        # Update plotting data
        self.plotting_dict["train_loss"].append(train_loss)
        self.plotting_dict["val_loss"].append(val_loss)

        # TensorBoard logging
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)

        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f'Parameters/{name}', param, epoch)

        # Check for improvement
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            print(
                f"Validation loss improved to {val_loss:.4f}. Saving model...")
        else:
            self.early_stop_counter += 1

        return val_loss, is_best

    def train(self, train_dataloader, val_dataloader, epochs=5, checkpoint_path=None):
        """Standard PyTorch training loop"""
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        for epoch in range(epochs):
            # Run training for one epoch
            train_loss = self._run_epoch(epoch, epochs, train_dataloader)

            # Handle validation and early stopping
            val_loss, is_best = self._handle_validation(
                epoch, val_dataloader, train_loss)

            # Save checkpoint if this is the best model
            if is_best:
                self.save_checkpoint(epoch, val_loss)

            # Early stopping check
            if self.early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        return val_loss

    def train_with_ray(self, train_dataloader, val_dataloader, epochs=5):
        """Ray Tune compatible training method"""
        import ray.train

        # Create a temporary directory for Ray checkpoints
        temp_checkpoint_dir = self.checkpoint_dir
        if temp_checkpoint_dir is None:
            temp_checkpoint_dir = os.path.abspath(
                os.path.join(os.getcwd(), 'ray_results'))
            os.makedirs(temp_checkpoint_dir, exist_ok=True)

        # Try to load checkpoint if exists through Ray
        start_epoch = 0
        checkpoint = ray.train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                print(f"Loading checkpoint from {checkpoint_dir}")
                checkpoint_dict = torch.load(
                    os.path.join(checkpoint_dir, "checkpoint.pt"))
                start_epoch = checkpoint_dict["epoch"] + 1
                self.model.load_state_dict(checkpoint_dict["model_state"])

        # Training loop
        for epoch in range(start_epoch, start_epoch + epochs):
            # Run training for one epoch (reusing the same logic)
            train_loss = self._run_epoch(
                epoch, start_epoch + epochs, train_dataloader)

            # Handle validation and early stopping
            val_loss, is_best = self._handle_validation(
                epoch, val_dataloader, train_loss)

            # Save Ray-specific checkpoint
            if is_best:
                checkpoint_path = os.path.join(
                    temp_checkpoint_dir, "checkpoint.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_loss": val_loss
                }, checkpoint_path)

            # Early stopping check
            if self.early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Report metrics to Ray
        try:
            print("Reporting metrics and checkpoint...")
            checkpoint = ray.train.Checkpoint.from_directory(
                temp_checkpoint_dir)
            ray.train.report({"val_loss": val_loss}, checkpoint=checkpoint)
        except Exception as e:
            print(f"Warning: Failed to report checkpoint: {e}")
            # Fallback to just reporting metrics without checkpoint
            try:
                ray.train.report({"val_loss": val_loss})
            except Exception as e:
                print(f"Warning: Failed to report metrics: {e}")

        return val_loss

    def evaluate(self, dataloader, mode="Evaluation"):
        """Evaluate the model on a dataset"""
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for inputs, targets, category in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                category = category.to(self.device)

                outputs = self.model(inputs, category)
                loss += self.criterion(outputs, targets.float()).item()

        loss /= len(dataloader)
        print(f"{mode} Loss: {loss:.4f}")
        print("-------------------------")
        return loss

    def plot(self):
        """Plot training and validation loss curves"""
        fig = plt.figure(figsize=(10, 6))
        plt.plot(self.plotting_dict["train_loss"], label="Train Loss")
        plt.plot(self.plotting_dict["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()

        self.writer.add_figure('Loss Curves', fig)
        plt.show()

        return fig

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()

    def save_checkpoint(self, epoch, val_loss):
        """Save a model checkpoint"""
        if not self.checkpoint_dir:
            return

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, checkpoint_path)

        # Also save as best model if this is the best validation loss
        if val_loss == self.best_val_loss:
            best_model_path = os.path.join(
                self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved: {best_model_path}")

        print(f"Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path):
        """Private method to load a checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("val_loss", float('inf'))
        print(f"Checkpoint loaded from {checkpoint_path}")
        return True

    def load_checkpoint(self, checkpoint_path=None):
        """Load a model checkpoint - if no path specified, load best model"""
        if checkpoint_path is None and self.checkpoint_dir:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, "best_model.pt")

        return self._load_checkpoint(checkpoint_path)


class Trainer_Advanced:
    """
    Advanced Trainer class for PyTorch
    """

    def __init__(self, model, criterion, optimizer, device=None, log_dir=None, checkpoint_dir=None,
                 early_stop_patience=5, logger=None, config=None):
        # Auto-detect device if not provided
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.plotting_dict = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "training_time": [],
        }
        self.early_stop_counter = 0
        self.early_stop_patience = early_stop_patience
        self.best_val_loss = float('inf')
        self.checkpoint_dir = checkpoint_dir
        self.config = config or {}  # Store configuration for reproducibility

        # Setup logging
        self.logger = logger or self._setup_logger()
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Model parameters: {self._count_parameters():,}")

        # Create checkpoint directory if it doesn't exist
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

        # TensorBoard setup
        if log_dir is None:
            current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
            log_dir = os.path.join('runs', current_time)

        self.writer = SummaryWriter(log_dir)
        self.logger.info(f"TensorBoard logs directory: {log_dir}")

        # Initialize metrics tracking
        self.metrics_history = {}

        # Log model architecture
        self._log_model_summary()

    def _setup_logger(self):
        """Set up a logger for the trainer"""
        logger = logging.getLogger("Trainer")
        logger.setLevel(logging.INFO)

        # Create handlers
        c_handler = logging.StreamHandler()
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_file = f"training_{current_time}.log"

        if self.checkpoint_dir:
            f_handler = logging.FileHandler(
                os.path.join(self.checkpoint_dir, log_file))
        else:
            os.makedirs('logs', exist_ok=True)
            f_handler = logging.FileHandler(os.path.join('logs', log_file))

        # Create formatters and add to handlers
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        c_format = logging.Formatter(format_str)
        f_format = logging.Formatter(format_str)
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        return logger

    def _log_model_summary(self):
        """Log model architecture summary"""
        self.logger.info(f"Model architecture: {type(self.model).__name__}")
        self.logger.info(f"Criterion: {type(self.criterion).__name__}")
        self.logger.info(f"Optimizer: {type(self.optimizer).__name__}")

        # Log model to TensorBoard
        dummy_size = next(self.model.parameters()).size()
        if len(dummy_size) >= 2:  # If tensor has at least 2 dimensions
            try:
                dummy_input = (
                    torch.zeros(1, dummy_size[1]).to(self.device),
                    # Assuming category input is 1D
                    torch.zeros(1, 1).to(self.device)
                )
                self.writer.add_graph(self.model, dummy_input)
            except Exception as e:
                self.logger.warning(
                    f"Failed to add model graph to TensorBoard: {e}")

    def _count_parameters(self):
        """Count number of trainable parameters in the model"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _run_epoch(self, epoch, epochs, train_dataloader, use_tqdm=True):
        """Run a single training epoch with progress bar"""
        self.model.train()
        total_loss = 0

        # Create progress bar
        loader = tqdm(train_dataloader,
                      desc=f"Epoch {epoch+1}/{epochs}", disable=not use_tqdm)

        for i, (inputs, targets, category) in enumerate(loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            category = category.to(self.device)

            outputs = self.model(inputs, category)
            loss = self.criterion(outputs, targets.float())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

            # Update progress bar
            if use_tqdm:
                loader.set_postfix(loss=f"{loss.item():.4f}")

            if i % 10 == 0:
                step = epoch * len(train_dataloader) + i
                self.writer.add_scalar('Batch/Loss', loss.item(), step)

        # Store learning rate
        current_lr = self._get_current_lr()
        self.plotting_dict["learning_rates"].append(current_lr)
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Calculate average training loss for the epoch
        train_loss = total_loss / len(train_dataloader)
        self.logger.info(
            f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, LR: {current_lr:.6f}")

        return train_loss

    def _get_current_lr(self):
        """Get current learning rate"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _handle_validation(self, epoch, val_dataloader, train_loss, use_tqdm=True):
        """Handle validation, early stopping, and checkpointing"""
        val_loss, metrics = self.evaluate(
            val_dataloader, "Validation", use_tqdm)

        # Update plotting data
        self.plotting_dict["train_loss"].append(train_loss)
        self.plotting_dict["val_loss"].append(val_loss)

        # Store metrics history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
            self.writer.add_scalar(f'Metrics/{key}', value, epoch)

        # TensorBoard logging
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)

        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f'Parameters/{name}', param, epoch)

        # Check for improvement
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            self.logger.info(
                f"Validation loss improved to {val_loss:.4f}. Saving model...")
        else:
            self.early_stop_counter += 1
            self.logger.info(
                f"Validation loss did not improve. Early stopping counter: {self.early_stop_counter}/{self.early_stop_patience}")

        return val_loss, metrics, is_best

    def train(self, train_dataloader, val_dataloader, epochs=5, checkpoint_path=None,
              scheduler=None, use_tqdm=True, ray_mode=False):
        """
        Standard PyTorch training loop

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: Number of training epochs
            checkpoint_path: Path to load a checkpoint from
            scheduler: Optional learning rate scheduler
            use_tqdm: Whether to display progress bars
            ray_mode: Whether to use Ray Tune for reporting

        Returns:
            Dict containing final metrics
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()

        # Save config info
        if self.checkpoint_dir:
            config_path = os.path.join(
                self.checkpoint_dir, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    "epochs": epochs,
                    "optimizer": str(self.optimizer),
                    "criterion": str(self.criterion),
                    "model": str(type(self.model).__name__),
                    **self.config
                }, f, indent=4)

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        for epoch in range(epochs):
            epoch_start = time.time()

            # Run training for one epoch
            train_loss = self._run_epoch(
                epoch, epochs, train_dataloader, use_tqdm)

            # Handle validation and early stopping
            val_loss, metrics, is_best = self._handle_validation(
                epoch, val_dataloader, train_loss, use_tqdm
            )

            # Update learning rate if scheduler exists
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                self.logger.info(
                    f"Learning rate updated to {self._get_current_lr():.6f}")

            # Record epoch time
            epoch_time = time.time() - epoch_start
            self.plotting_dict["training_time"].append(epoch_time)
            self.logger.info(f"Epoch time: {epoch_time:.2f}s")

            # Save checkpoint if this is the best model
            if is_best:
                self.save_checkpoint(epoch, val_loss, metrics)

            # Ray tune reporting
            if ray_mode:
                self._report_to_ray(val_loss, metrics)

            # Early stopping check
            if self.early_stop_counter >= self.early_stop_patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch+1} epochs")
                break

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")

        # Final evaluation on validation set
        self.logger.info("Performing final evaluation with best model...")
        self.load_checkpoint()  # Load best model
        final_val_loss, final_metrics = self.evaluate(
            val_dataloader, "Final Validation", use_tqdm)

        return {
            "train_loss": train_loss,
            "val_loss": final_val_loss,
            "training_time": total_time,
            **final_metrics
        }

    def _report_to_ray(self, val_loss, metrics=None):
        """Report metrics to Ray Tune"""
        try:
            import ray.train

            # Create a temporary directory for checkpoints if not provided
            temp_checkpoint_dir = self.checkpoint_dir
            if temp_checkpoint_dir is None:
                temp_checkpoint_dir = os.path.abspath(
                    os.path.join(os.getcwd(), 'ray_results'))
                os.makedirs(temp_checkpoint_dir, exist_ok=True)

            # Prepare metrics to report
            report_metrics = {"val_loss": val_loss}
            if metrics:
                report_metrics.update(metrics)

            checkpoint = ray.train.Checkpoint.from_directory(
                temp_checkpoint_dir)
            ray.train.report(report_metrics, checkpoint=checkpoint)
            self.logger.info(f"Reported metrics to Ray: {report_metrics}")
        except Exception as e:
            self.logger.warning(f"Warning: Failed to report to Ray: {e}")
            # Fallback to just reporting metrics without checkpoint
            try:
                ray.train.report({"val_loss": val_loss})
            except Exception as e:
                self.logger.warning(f"Warning: Failed to report metrics: {e}")

    def train_with_ray(self, train_dataloader, val_dataloader, epochs=5):
        """Ray Tune compatible training method"""
        return self.train(train_dataloader, val_dataloader, epochs, ray_mode=True)

    def evaluate(self, dataloader, mode="Evaluation", use_tqdm=False):
        """
        Evaluate the model on a dataset

        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []

        loader = tqdm(dataloader, desc=f"{mode}", disable=not use_tqdm)

        with torch.no_grad():
            for inputs, targets, category in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                category = category.to(self.device)

                outputs = self.model(inputs, category)
                loss = self.criterion(outputs, targets.float()).item()
                total_loss += loss

                # Store predictions and targets for metrics
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

                if use_tqdm:
                    loader.set_postfix(loss=f"{loss:.4f}")

        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        self.logger.info(f"{mode} Loss: {avg_loss:.4f}")

        # Calculate additional metrics if available
        metrics = self._calculate_metrics(
            torch.cat(all_outputs), torch.cat(all_targets))
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"{mode} Metrics: {metrics_str}")

        return avg_loss, metrics

    def _calculate_metrics(self, outputs, targets):
        """Calculate additional metrics beyond loss

        Can be extended for specific metrics like MSE, MAE, etc.
        """
        metrics = {}

        # Mean Squared Error
        mse = torch.mean((outputs - targets) ** 2).item()
        metrics['mse'] = mse

        # Mean Absolute Error
        mae = torch.mean(torch.abs(outputs - targets)).item()
        metrics['mae'] = mae

        # R2 score (coefficient of determination)
        try:
            ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
            ss_res = torch.sum((targets - outputs) ** 2).item()
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            metrics['r2'] = r2
        except:
            # Skip R2 if there's an issue calculating it
            pass

        return metrics

    def plot(self, save_path=None):
        """Plot training and validation loss curves plus other metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Plot loss
        axes[0, 0].plot(self.plotting_dict["train_loss"], label="Train Loss")
        axes[0, 0].plot(self.plotting_dict["val_loss"],
                        label="Validation Loss")
        axes[0, 0].set_xlabel("Epochs")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()

        # Plot learning rate
        if self.plotting_dict["learning_rates"]:
            axes[0, 1].plot(self.plotting_dict["learning_rates"])
            axes[0, 1].set_xlabel("Epochs")
            axes[0, 1].set_ylabel("Learning Rate")
            axes[0, 1].set_title("Learning Rate Schedule")
            if len(self.plotting_dict["learning_rates"]) > 1:
                axes[0, 1].set_yscale('log')

        # Plot other metrics if available
        if self.metrics_history:
            for i, (metric_name, values) in enumerate(self.metrics_history.items()):
                if i < 2:  # Only plot up to 2 metrics
                    ax = axes[1, i]
                    ax.plot(values)
                    ax.set_xlabel("Epochs")
                    ax.set_ylabel(metric_name)
                    ax.set_title(f"{metric_name} during Training")

        # Plot training time
        if len(self.plotting_dict["training_time"]) > 0:
            # If we don't have metrics to plot in the bottom right, use it for time
            if len(self.metrics_history) < 2:
                ax = axes[1, 1] if len(
                    self.metrics_history) < 1 else axes[1, 1]
                ax.plot(self.plotting_dict["training_time"])
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Time (s)")
                ax.set_title("Training Time per Epoch")

        plt.tight_layout()

        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Training plots saved to {save_path}")

        # Add figure to TensorBoard
        self.writer.add_figure('Training Plots', fig)

        plt.show()
        return fig

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()

        # Close all handlers for logger
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    def save_checkpoint(self, epoch, val_loss, metrics=None):
        """Save a model checkpoint"""
        if not self.checkpoint_dir:
            return

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "metrics": metrics or {},
            "config": self.config,
        }
        torch.save(checkpoint, checkpoint_path)

        # Also save as best model if this is the best validation loss
        if val_loss == self.best_val_loss:
            best_model_path = os.path.join(
                self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"Best model saved: {best_model_path}")

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path):
        """Private method to load a checkpoint"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"No checkpoint found at {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load state dicts based on keys available in the checkpoint
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.best_val_loss = checkpoint.get("val_loss", float('inf'))
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return True

    def load_checkpoint(self, checkpoint_path=None):
        """Load a model checkpoint - if no path specified, load best model"""
        if checkpoint_path is None and self.checkpoint_dir:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, "best_model.pt")

        return self._load_checkpoint(checkpoint_path)

    def predict(self, dataloader, return_targets=False):
        """
        Generate predictions using the trained model

        Args:
            dataloader: DataLoader for prediction
            return_targets: Whether to also return targets

        Returns:
            Numpy array of predictions (and targets if requested)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets, category in tqdm(dataloader, desc="Predicting"):
                inputs = inputs.to(self.device)
                category = category.to(self.device)

                outputs = self.model(inputs, category)
                all_predictions.append(outputs.cpu().numpy())

                if return_targets:
                    all_targets.append(targets.numpy())

        predictions = np.vstack(all_predictions)

        if return_targets:
            targets = np.vstack(all_targets)
            return predictions, targets

        return predictions

    def export_model(self, path=None, format="pt"):
        """
        Export the trained model in a specific format

        Args:
            path: Path to save the model (default: checkpoint_dir/model.{format})
            format: Format to save as ('pt', 'onnx', 'script')
        """
        if path is None and self.checkpoint_dir:
            path = os.path.join(self.checkpoint_dir, f"model.{format}")
        elif path is None:
            path = f"model.{format}"

        if format == "pt":
            torch.save(self.model.state_dict(), path)
            self.logger.info(f"Model saved as PyTorch state dict: {path}")

        elif format == "script":
            # Export as TorchScript
            try:
                dummy_size = next(self.model.parameters()).size()
                dummy_input = (
                    torch.zeros(1, dummy_size[1]).to(self.device),
                    torch.zeros(1, 1).to(self.device)
                )
                script_model = torch.jit.trace(self.model, dummy_input)
                script_model.save(path)
                self.logger.info(f"Model saved as TorchScript: {path}")
            except Exception as e:
                self.logger.error(f"Failed to export as TorchScript: {e}")

        elif format == "onnx":
            # Export as ONNX
            try:
                import onnx
                dummy_size = next(self.model.parameters()).size()
                dummy_input = (
                    torch.zeros(1, dummy_size[1]).to(self.device),
                    torch.zeros(1, 1).to(self.device)
                )
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    path,
                    export_params=True,
                    opset_version=11,
                    input_names=['input', 'category'],
                    output_names=['output'],
                )
                self.logger.info(f"Model exported to ONNX: {path}")
            except Exception as e:
                self.logger.error(f"Failed to export as ONNX: {e}")

        else:
            self.logger.error(f"Unsupported export format: {format}")
