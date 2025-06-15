"""
Configuration module for T5 distillation
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any
import logging

@dataclass
class ModelConfig:
    """Configuration for teacher and student models"""
    teacher_model_name: str = "t5-large"
    student_model_name: str = "t5-small"
    max_input_length: int = 512
    max_target_length: int = 128
    cache_dir: str = "./models_cache"

@dataclass
class DataConfig:
    """Configuration for dataset"""
    dataset_name: str = "cnn_dailymail"
    dataset_version: str = "3.0.0"
    train_size: int = 1000  # Set to None for full dataset
    val_size: int = 200    # Set to None for full validation
    test_size: int = 100   # Set to None for full test
    cache_dir: str = "./data_cache"
    num_workers: int = 4

@dataclass
class DistillationConfig:
    """Configuration for distillation process"""
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for ground truth loss
    kl_loss_weight: float = 1.0

@dataclass
class TrainingConfig:
    """Configuration for training"""
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    output_dir: str = "./distilled_model"
    logs_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"

@dataclass
class SystemConfig:
    """System configuration"""
    device: str = "cuda"  # cuda, cpu, mps
    mixed_precision: bool = True
    dry_run: bool = False  # Set to True for testing without actual training
    seed: int = 42
    log_level: str = "INFO"

class ConfigManager:
    """Manages all configuration aspects"""
    
    def __init__(self, config_path: str = "distillation_config.json"):
        self.config_path = config_path
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.distillation_config = DistillationConfig()
        self.training_config = TrainingConfig()
        self.system_config = SystemConfig()
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for configuration"""
        logger = logging.getLogger("ConfigManager")
        logger.setLevel(getattr(logging, self.system_config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_config(self) -> None:
        """Load configuration from JSON file"""
        if os.path.exists(self.config_path):
            self.logger.info(f"Loading configuration from {self.config_path}")
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Update configurations
            if 'model_config' in config_dict:
                self.model_config = ModelConfig(**config_dict['model_config'])
            if 'data_config' in config_dict:
                self.data_config = DataConfig(**config_dict['data_config'])
            if 'distillation_config' in config_dict:
                self.distillation_config = DistillationConfig(**config_dict['distillation_config'])
            if 'training_config' in config_dict:
                self.training_config = TrainingConfig(**config_dict['training_config'])
            if 'system_config' in config_dict:
                self.system_config = SystemConfig(**config_dict['system_config'])
                
            self.logger.info("Configuration loaded successfully")
        else:
            self.logger.info(f"No config file found at {self.config_path}, using defaults")
    
    def save_config(self) -> None:
        """Save current configuration to JSON file"""
        config_dict = {
            'model_config': asdict(self.model_config),
            'data_config': asdict(self.data_config),
            'distillation_config': asdict(self.distillation_config),
            'training_config': asdict(self.training_config),
            'system_config': asdict(self.system_config)
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Configuration saved to {self.config_path}")
    
    def create_directories(self) -> None:
        """Create necessary directories"""
        dirs_to_create = [
            self.model_config.cache_dir,
            self.data_config.cache_dir,
            self.training_config.output_dir,
            self.training_config.logs_dir,
            self.training_config.checkpoint_dir
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations as dictionary"""
        return {
            'model_config': self.model_config,
            'data_config': self.data_config,
            'distillation_config': self.distillation_config,
            'training_config': self.training_config,
            'system_config': self.system_config
        }
    
    def print_config_summary(self) -> None:
        """Print configuration summary"""
        self.logger.info("=" * 50)
        self.logger.info("CONFIGURATION SUMMARY")
        self.logger.info("=" * 50)
        
        configs = self.get_all_configs()
        for config_name, config_obj in configs.items():
            self.logger.info(f"\n{config_name.upper()}:")
            for key, value in asdict(config_obj).items():
                self.logger.info(f"  {key}: {value}")