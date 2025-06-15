"""
Main training module for T5 distillation
"""
import logging
import os
import time
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from config import ConfigManager
from model import ModelManager
from distillation import DistillationTrainer
from dataloader import DataLoaderManager


class T5DistillationTrainer:
    """Main trainer for T5 distillation"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.configs = config_manager.get_all_configs()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize components
        self.model_manager = ModelManager(
            self.configs['model_config'], 
            self.configs['system_config']
        )
        
        self.data_manager = DataLoaderManager(
            self.configs['data_config'],
            self.configs['system_config']
        )
        
        self.distillation_trainer = DistillationTrainer(
            self.configs['distillation_config'],
            self.configs['system_config']
        )
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.logger.info("T5DistillationTrainer initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup main trainer logger"""
        logger = logging.getLogger("T5DistillationTrainer")
        logger.setLevel(getattr(logging, self.configs['system_config'].log_level))
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = os.path.join(
                self.configs['training_config'].logs_dir, 
                'training.log'
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def setup_training(self) -> None:
        """Setup all training components"""
        self.logger.info("Setting up training components...")
        
        # Load models
        self.logger.info("Loading models...")
        teacher_model = self.model_manager.load_teacher_model()
        student_model = self.model_manager.load_student_model()
        
        # Print model summary
        self.model_manager.print_model_summary()
        
        # Setup data
        self.logger.info("Setting up data...")
        tokenizer = self.data_manager.setup_tokenizer(
            self.configs['model_config'].teacher_model_name
        )
        
        # Load dataset
        train_data, val_data, test_data = self.data_manager.load_dataset()
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = self.data_manager.create_datasets(
            train_data, val_data, test_data,
            self.configs['model_config'].max_input_length,
            self.configs['model_config'].max_target_length
        )
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = self.data_manager.create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            self.configs['training_config'].batch_size
        )
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Setup mixed precision if enabled
        if self.configs['system_config'].mixed_precision and not self.configs['system_config'].dry_run:
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Mixed precision training enabled")
        
        self.logger.info("Training setup completed!")
    
    def _setup_optimizer_and_scheduler(self) -> None:
        """Setup optimizer and learning rate scheduler"""
        
        if self.configs['system_config'].dry_run:
            # Create dummy optimizer for dry run
            dummy_param = torch.nn.Parameter(torch.tensor(0.0))
            self.optimizer = AdamW([dummy_param], lr=self.configs['training_config'].learning_rate)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=10,
                num_training_steps=100
            )
            self.logger.info("DRY RUN MODE: Created dummy optimizer and scheduler")
            return
        
        # Real optimizer setup
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model_manager.student_model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model_manager.student_model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.configs['training_config'].learning_rate,
            eps=1e-8
        )
        
        # Calculate total training steps
        total_steps = (
            len(self.train_loader) * self.configs['training_config'].num_epochs
        ) // self.configs['training_config'].gradient_accumulation_steps
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.configs['training_config'].warmup_steps,
            num_training_steps=total_steps
        )
        
        self.logger.info(f"Optimizer and scheduler setup completed. Total steps: {total_steps}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.logger.info(f"Starting epoch {epoch + 1}/{self.configs['training_config'].num_epochs}")
        
        if self.configs['system_config'].dry_run:
            return self._dry_run_epoch(epoch, mode='train')
        
        self.model_manager.student_model.train()
        epoch_metrics = []
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch + 1}", 
            disable=self.configs['system_config'].dry_run
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self.model_manager.move_batch_to_device(batch)
            
            # Get model outputs
            teacher_outputs = self.model_manager.get_teacher_outputs(batch)
            student_outputs = self.model_manager.get_student_outputs(batch)
            
            # Calculate loss
            loss_dict = self.distillation_trainer.train_step(
                student_outputs, teacher_outputs, batch['labels']
            )
            
            loss = loss_dict['total_loss']
            
            # Backward pass with gradient accumulation
            if self.scaler is not None:
                # Mixed precision
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.configs['training_config'].gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model_manager.student_model.parameters(),
                        self.configs['training_config'].max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            else:
                # Standard precision
                loss.backward()
                
                if (batch_idx + 1) % self.configs['training_config'].gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_manager.student_model.parameters(),
                        self.configs['training_config'].max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            current_lr = self.scheduler.get_last_lr()[0]
            self.distillation_trainer.update_training_metrics(loss_dict, current_lr)
            
            self.global_step += 1
            
            # Logging
            if self.global_step % self.configs['training_config'].logging_steps == 0:
                avg_metrics = self.distillation_trainer.get_average_metrics(
                    self.distillation_trainer.training_metrics, 
                    self.configs['training_config'].logging_steps
                )
                self.distillation_trainer.log_metrics(avg_metrics, self.global_step, "train")
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{avg_metrics['total_loss']:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
            
            # Validation
            if self.global_step % self.configs['training_config'].eval_steps == 0:
                val_metrics = self.validate()
                self.model_manager.student_model.train()  # Back to training mode
            
            # Save checkpoint
            if self.global_step % self.configs['training_config'].save_steps == 0:
                self.save_checkpoint(f"checkpoint-step-{self.global_step}")
        
        # Get epoch metrics
        epoch_metrics = self.distillation_trainer.get_average_metrics(
            self.distillation_trainer.training_metrics
        )
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.logger.info("Running validation...")
        
        if self.configs['system_config'].dry_run:
            return self._dry_run_epoch(0, mode='val')
        
        self.model_manager.student_model.eval()
        
        val_metrics_epoch = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", disable=False):
                # Move batch to device
                batch = self.model_manager.move_batch_to_device(batch)
                
                # Get model outputs
                teacher_outputs = self.model_manager.get_teacher_outputs(batch)
                student_outputs = self.model_manager.get_student_outputs(batch)
                
                # Calculate validation metrics
                loss_dict = self.distillation_trainer.validate_step(
                    student_outputs, teacher_outputs, batch['labels']
                )
                
                self.distillation_trainer.update_validation_metrics(loss_dict)
        
        # Get average validation metrics
        val_metrics = self.distillation_trainer.get_average_metrics(
            self.distillation_trainer.validation_metrics,
            len(self.val_loader)
        )
        
        self.distillation_trainer.log_metrics(val_metrics, self.global_step, "validation")
        
        # Check if this is the best model
        current_val_loss = val_metrics['total_loss']
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.save_best_model()
            self.logger.info(f"New best validation loss: {self.best_val_loss:.6f}")
        
        return val_metrics
    
    def _dry_run_epoch(self, epoch: int, mode: str = 'train') -> Dict[str, float]:
        """Simulate an epoch for dry run mode"""
        self.logger.info(f"DRY RUN MODE: Simulating {mode} epoch {epoch + 1}")
        
        # Simulate some batches
        num_batches = 5 if mode == 'train' else 2
        
        for i in range(num_batches):
            time.sleep(0.1)  # Simulate processing time
            
            # Simulate loss values
            dummy_loss = {
                'total_loss': torch.tensor(1.0 - i * 0.1),
                'distillation_loss': torch.tensor(0.7 - i * 0.05),
                'ground_truth_loss': torch.tensor(0.3 - i * 0.02)
            }
            
            if mode == 'train':
                self.distillation_trainer.update_training_metrics(dummy_loss, 5e-5)
                self.global_step += 1
            else:
                dummy_loss['perplexity'] = torch.tensor(2.5 - i * 0.1)
                self.distillation_trainer.update_validation_metrics(dummy_loss)
        
        # Return dummy metrics
        return {
            'total_loss': 0.8,
            'distillation_loss': 0.6,
            'ground_truth_loss': 0.2,
            'learning_rate': 5e-5
        }
    
    def train(self) -> None:
        """Main training loop"""
        self.logger.info("Starting training...")
        
        start_time = time.time()
        
        for epoch in range(self.configs['training_config'].num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch summary
            self.logger.info(f"Epoch {epoch + 1} completed:")
            self.logger.info(f"  Train Loss: {train_metrics['total_loss']:.6f}")
            self.logger.info(f"  Val Loss: {val_metrics['total_loss']:.6f}")
            if 'perplexity' in val_metrics:
                self.logger.info(f"  Val Perplexity: {val_metrics['perplexity']:.4f}")
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model and metrics
        self.save_final_model()
        self.save_training_metrics()
    
    def save_checkpoint(self, checkpoint_name: str) -> None:
        """Save training checkpoint"""
        if self.configs['system_config'].dry_run:
            self.logger.info(f"DRY RUN MODE: Skipping checkpoint save {checkpoint_name}")
            return
        
        checkpoint_path = os.path.join(
            self.configs['training_config'].checkpoint_dir, 
            checkpoint_name
        )
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model_manager.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config_manager.get_all_configs()
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, f"{checkpoint_path}.pt")
        self.logger.info(f"Checkpoint saved: {checkpoint_path}.pt")
    
    def save_best_model(self) -> None:
        """Save the best model"""
        best_model_path = os.path.join(
            self.configs['training_config'].output_dir, 
            "best_model"
        )
        self.model_manager.save_student_model(best_model_path)
    
    def save_final_model(self) -> None:
        """Save the final model"""
        final_model_path = os.path.join(
            self.configs['training_config'].output_dir, 
            "final_model"
        )
        self.model_manager.save_student_model(final_model_path)
    
    def save_training_metrics(self) -> None:
        """Save training metrics"""
        metrics_path = os.path.join(
            self.configs['training_config'].logs_dir,
            "training_metrics.json"
        )
        self.distillation_trainer.save_metrics(metrics_path)
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        self.model_manager.cleanup_models()
        self.logger.info("Cleanup completed")