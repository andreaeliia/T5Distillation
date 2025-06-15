"""
Distillation module for T5 model distillation
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from config import DistillationConfig, SystemConfig

class DistillationLoss(nn.Module):
    """Custom loss function for knowledge distillation"""
    
    def __init__(self, distillation_config: DistillationConfig):
        super().__init__()
        self.temperature = distillation_config.temperature
        self.alpha = distillation_config.alpha
        self.beta = distillation_config.beta
        self.kl_loss_weight = distillation_config.kl_loss_weight
        
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"DistillationLoss initialized with T={self.temperature}, "
                        f"α={self.alpha}, β={self.beta}")
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate distillation loss
        
        Args:
            student_logits: Student model logits [batch_size, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch_size, seq_len, vocab_size]
            labels: Ground truth labels [batch_size, seq_len]
        
        Returns:
            Dictionary with loss components
        """
        
        # Reshape logits and labels for loss calculation
        student_logits_flat = student_logits.view(-1, student_logits.size(-1))
        teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
        labels_flat = labels.view(-1)
        
        # Create mask for valid positions (not -100)
        mask = (labels_flat != -100)
        
        if mask.sum() == 0:
            # No valid positions
            return {
                'total_loss': torch.tensor(0.0, device=student_logits.device),
                'distillation_loss': torch.tensor(0.0, device=student_logits.device),
                'ground_truth_loss': torch.tensor(0.0, device=student_logits.device)
            }
        
        # Apply mask
        student_logits_masked = student_logits_flat[mask]
        teacher_logits_masked = teacher_logits_flat[mask]
        labels_masked = labels_flat[mask]
        
        # Calculate soft targets (teacher predictions)
        teacher_probs = F.softmax(teacher_logits_masked / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits_masked / self.temperature, dim=-1)
        
        # Distillation loss (KL divergence)
        distillation_loss = self.kl_div_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Ground truth loss (standard cross-entropy)
        ground_truth_loss = self.ce_loss(student_logits_masked, labels_masked)
        
        # Combined loss
        total_loss = (self.alpha * distillation_loss + 
                     self.beta * ground_truth_loss) * self.kl_loss_weight
        
        return {
            'total_loss': total_loss,
            'distillation_loss': distillation_loss,
            'ground_truth_loss': ground_truth_loss
        }

class DistillationTrainer:
    """Trainer for knowledge distillation"""
    
    def __init__(
        self,
        distillation_config: DistillationConfig,
        system_config: SystemConfig
    ):
        self.distillation_config = distillation_config
        self.system_config = system_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.loss_fn = DistillationLoss(distillation_config)
        self.device = torch.device(system_config.device)
        
        # Metrics tracking
        self.training_metrics = {
            'total_loss': [],
            'distillation_loss': [],
            'ground_truth_loss': [],
            'learning_rate': []
        }
        
        self.validation_metrics = {
            'total_loss': [],
            'distillation_loss': [],
            'ground_truth_loss': [],
            'perplexity': []
        }
        
        self.logger.info("DistillationTrainer initialized")
    
    def train_step(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one training step
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs  
            labels: Ground truth labels
            
        Returns:
            Dictionary with loss components
        """
        
        if self.system_config.dry_run:
            # Return dummy losses for dry run
            return {
                'total_loss': torch.tensor(1.0, device=self.device),
                'distillation_loss': torch.tensor(0.7, device=self.device),
                'ground_truth_loss': torch.tensor(0.3, device=self.device)
            }
        
        student_logits = student_outputs['logits']
        teacher_logits = teacher_outputs['logits']
        
        # Calculate distillation loss
        loss_dict = self.loss_fn(student_logits, teacher_logits, labels)
        
        return loss_dict
    
    def validate_step(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one validation step
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            labels: Ground truth labels
            
        Returns:
            Dictionary with loss components and metrics
        """
        
        with torch.no_grad():
            loss_dict = self.train_step(student_outputs, teacher_outputs, labels)
            
            if not self.system_config.dry_run:
                # Calculate perplexity
                student_loss = student_outputs.get('loss')
                if student_loss is not None:
                    perplexity = torch.exp(student_loss)
                    loss_dict['perplexity'] = perplexity
                else:
                    loss_dict['perplexity'] = torch.tensor(float('inf'), device=self.device)
            else:
                loss_dict['perplexity'] = torch.tensor(2.5, device=self.device)
        
        return loss_dict
    
    def update_training_metrics(self, loss_dict: Dict[str, torch.Tensor], lr: float) -> None:
        """Update training metrics"""
        self.training_metrics['total_loss'].append(loss_dict['total_loss'].item())
        self.training_metrics['distillation_loss'].append(loss_dict['distillation_loss'].item())
        self.training_metrics['ground_truth_loss'].append(loss_dict['ground_truth_loss'].item())
        self.training_metrics['learning_rate'].append(lr)
    
    def update_validation_metrics(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        """Update validation metrics"""
        self.validation_metrics['total_loss'].append(loss_dict['total_loss'].item())
        self.validation_metrics['distillation_loss'].append(loss_dict['distillation_loss'].item())
        self.validation_metrics['ground_truth_loss'].append(loss_dict['ground_truth_loss'].item())
        
        if 'perplexity' in loss_dict:
            self.validation_metrics['perplexity'].append(loss_dict['perplexity'].item())
    
    def get_average_metrics(self, metrics_dict: Dict[str, list], last_n: int = None) -> Dict[str, float]:
        """Get average metrics over last n steps"""
        avg_metrics = {}
        
        for key, values in metrics_dict.items():
            if values:
                if last_n is not None:
                    values_subset = values[-last_n:] if len(values) >= last_n else values
                else:
                    values_subset = values
                avg_metrics[key] = sum(values_subset) / len(values_subset)
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics
    
    def log_metrics(self, metrics: Dict[str, float], step: int, mode: str = "train") -> None:
        """Log metrics"""
        self.logger.info(f"{mode.upper()} - Step {step}:")
        for key, value in metrics.items():
            if key == 'perplexity':
                self.logger.info(f"  {key}: {value:.4f}")
            elif 'loss' in key:
                self.logger.info(f"  {key}: {value:.6f}")
            elif key == 'learning_rate':
                self.logger.info(f"  {key}: {value:.2e}")
            else:
                self.logger.info(f"  {key}: {value:.4f}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.training_metrics = {key: [] for key in self.training_metrics.keys()}
        self.validation_metrics = {key: [] for key in self.validation_metrics.keys()}
        self.logger.info("Metrics reset")
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics"""
        return {
            'training': self.get_average_metrics(self.training_metrics),
            'validation': self.get_average_metrics(self.validation_metrics)
        }
    
    def save_metrics(self, save_path: str) -> None:
        """Save metrics to file"""
        import json
        
        metrics_data = {
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'summary': self.get_metrics_summary()
        }
        
        if not self.system_config.dry_run:
            with open(save_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            self.logger.info(f"Metrics saved to {save_path}")
        else:
            self.logger.info(f"DRY RUN MODE: Skipping metrics save to {save_path}")

class HiddenStateAligner:
    """Aligns hidden states between teacher and student models of different sizes"""
    
    def __init__(self, teacher_dim: int, student_dim: int, device: torch.device):
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.device = device
        
        # Create linear projection layer if dimensions don't match
        if teacher_dim != student_dim:
            self.projection = nn.Linear(student_dim, teacher_dim).to(device)
        else:
            self.projection = None
            
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"HiddenStateAligner: {student_dim} -> {teacher_dim}")
    
    def align_hidden_states(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align hidden states between teacher and student
        
        Args:
            student_hidden: Student hidden states
            teacher_hidden: Teacher hidden states
            
        Returns:
            Aligned student and teacher hidden states
        """
        
        if self.projection is not None:
            # Project student hidden states to teacher dimension
            student_aligned = self.projection(student_hidden)
        else:
            student_aligned = student_hidden
        
        # Ensure same sequence length by padding or truncating
        if student_aligned.size(1) != teacher_hidden.size(1):
            min_len = min(student_aligned.size(1), teacher_hidden.size(1))
            student_aligned = student_aligned[:, :min_len, :]
            teacher_aligned = teacher_hidden[:, :min_len, :]
        else:
            teacher_aligned = teacher_hidden
        
        return student_aligned, teacher_aligned
    
    def hidden_state_loss(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Calculate MSE loss between aligned hidden states"""
        
        student_aligned, teacher_aligned = self.align_hidden_states(
            student_hidden, teacher_hidden
        )
        
        return F.mse_loss(student_aligned, teacher_aligned)