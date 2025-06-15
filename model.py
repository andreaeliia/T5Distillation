"""
Model management module for T5 distillation
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from transformers import T5ForConditionalGeneration, T5Config
from config import ModelConfig, SystemConfig

class ModelManager:
    """Manager for teacher and student models"""
    
    def __init__(self, model_config: ModelConfig, system_config: SystemConfig):
        self.model_config = model_config
        self.system_config = system_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.teacher_model = None
        self.student_model = None
        self.device = torch.device(system_config.device)
        
        self.logger.info(f"ModelManager initialized with device: {self.device}")
    
    def load_teacher_model(self) -> T5ForConditionalGeneration:
        """Load teacher model (T5-Large)"""
        self.logger.info(f"Loading teacher model: {self.model_config.teacher_model_name}")
        
        if self.system_config.dry_run:
            self.logger.info("DRY RUN MODE: Creating minimal teacher model")
            # Create a minimal config for dry run
            config = T5Config.from_pretrained(self.model_config.teacher_model_name)
            config.d_model = 64  # Much smaller
            config.d_ff = 128
            config.num_heads = 2
            config.num_layers = 2
            config.num_decoder_layers = 2
            #commento meme
            
            self.teacher_model = T5ForConditionalGeneration(config)
            self.logger.info("DRY RUN: Minimal teacher model created")
        else:
            self.teacher_model = T5ForConditionalGeneration.from_pretrained(
                self.model_config.teacher_model_name,
                cache_dir=self.model_config.cache_dir
            )
        
        # Move to device and set to eval mode
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        # Freeze teacher model parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        num_params = sum(p.numel() for p in self.teacher_model.parameters())
        self.logger.info(f"Teacher model loaded with {num_params:,} parameters")
        
        return self.teacher_model
    
    def load_student_model(self) -> T5ForConditionalGeneration:
        """Load student model (T5-Small)"""
        self.logger.info(f"Loading student model: {self.model_config.student_model_name}")
        
        if self.system_config.dry_run:
            self.logger.info("DRY RUN MODE: Creating minimal student model")
            # Create a minimal config for dry run
            config = T5Config.from_pretrained(self.model_config.student_model_name)
            config.d_model = 32  # Even smaller than teacher
            config.d_ff = 64
            config.num_heads = 2
            config.num_layers = 1
            config.num_decoder_layers = 1
            
            self.student_model = T5ForConditionalGeneration(config)
            self.logger.info("DRY RUN: Minimal student model created")
        else:
            self.student_model = T5ForConditionalGeneration.from_pretrained(
                self.model_config.student_model_name,
                cache_dir=self.model_config.cache_dir
            )
        
        # Move to device and set to training mode
        self.student_model.to(self.device)
        self.student_model.train()
        
        num_params = sum(p.numel() for p in self.student_model.parameters())
        trainable_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        
        self.logger.info(f"Student model loaded with {num_params:,} parameters "
                        f"({trainable_params:,} trainable)")
        
        return self.student_model
    
    def get_model_outputs(
        self,
        model: T5ForConditionalGeneration,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Get model outputs including logits and hidden states"""
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )
        
        return {
            'logits': outputs.logits,
            'loss': outputs.loss if labels is not None else None,
            'encoder_hidden_states': outputs.encoder_hidden_states,
            'decoder_hidden_states': outputs.decoder_hidden_states,
            'encoder_attentions': outputs.encoder_attentions,
            'decoder_attentions': outputs.decoder_attentions
        }
    
    def get_teacher_outputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get teacher model outputs"""
        if self.teacher_model is None:
            raise ValueError("Teacher model not loaded")
        
        with torch.no_grad():
            return self.get_model_outputs(
                self.teacher_model,
                batch['input_ids'],
                batch['attention_mask'],
                batch['decoder_input_ids'],
                batch['decoder_attention_mask'],
                batch['labels']
            )
    
    def get_student_outputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get student model outputs"""
        if self.student_model is None:
            raise ValueError("Student model not loaded")
        
        return self.get_model_outputs(
            self.student_model,
            batch['input_ids'],
            batch['attention_mask'],
            batch['decoder_input_ids'],
            batch['decoder_attention_mask'],
            batch['labels']
        )
    
    def save_student_model(self, save_path: str) -> None:
        """Save the distilled student model"""
        if self.student_model is None:
            raise ValueError("Student model not loaded")
        
        self.logger.info(f"Saving student model to {save_path}")
        
        if not self.system_config.dry_run:
            self.student_model.save_pretrained(save_path)
            self.logger.info("Student model saved successfully")
        else:
            self.logger.info("DRY RUN MODE: Skipping actual model save")
    
    def get_model_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary of model parameters"""
        summary = {}
        
        if self.teacher_model is not None:
            teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
            teacher_trainable = sum(p.numel() for p in self.teacher_model.parameters() if p.requires_grad)
            summary['teacher'] = {
                'total_params': teacher_params,
                'trainable_params': teacher_trainable
            }
        
        if self.student_model is not None:
            student_params = sum(p.numel() for p in self.student_model.parameters())
            student_trainable = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
            summary['student'] = {
                'total_params': student_params,
                'trainable_params': student_trainable
            }
        
        return summary
    
    def print_model_summary(self) -> None:
        """Print model summary"""
        summary = self.get_model_summary()
        
        self.logger.info("=" * 50)
        self.logger.info("MODEL SUMMARY")
        self.logger.info("=" * 50)
        
        for model_type, params in summary.items():
            self.logger.info(f"\n{model_type.upper()} MODEL:")
            self.logger.info(f"  Total parameters: {params['total_params']:,}")
            self.logger.info(f"  Trainable parameters: {params['trainable_params']:,}")
            
            if model_type == 'student' and 'teacher' in summary:
                compression_ratio = summary['teacher']['total_params'] / params['total_params']
                self.logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
    
    def move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to the correct device"""
        return {key: value.to(self.device) for key, value in batch.items()}
    
    def cleanup_models(self) -> None:
        """Clean up models from memory"""
        self.logger.info("Cleaning up models from memory")
        
        if self.teacher_model is not None:
            del self.teacher_model
            self.teacher_model = None
        
        if self.student_model is not None:
            del self.student_model
            self.student_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Models cleaned up successfully")