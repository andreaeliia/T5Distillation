"""
Main script for T5 distillation
"""
import argparse
import logging
import os
import sys
import torch
import random
import numpy as np
from config import ConfigManager
from trainer import T5DistillationTrainer

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_device(config_manager: ConfigManager) -> str:
    """Setup and validate device"""
    system_config = config_manager.system_config
    
    if system_config.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        system_config.device = "cpu"
    elif system_config.device == "mps" and not torch.backends.mps.is_available():
        logging.warning("MPS not available, falling back to CPU")
        system_config.device = "cpu"
    
    device = torch.device(system_config.device)
    logging.info(f"Using device: {device}")
    
    if system_config.device == "cuda":
        logging.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logging.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return system_config.device

def main():
    parser = argparse.ArgumentParser(description="T5 Model Distillation")
    parser.add_argument(
        "--config", 
        type=str, 
        default="distillation_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Run in dry-run mode (test functionality without actual training)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a default configuration file and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize configuration manager
    config_manager = ConfigManager(args.config)
    
    # Create config file if requested
    if args.create_config:
        config_manager.save_config()
        print(f"Default configuration saved to {args.config}")
        return
    
    # Load existing config
    config_manager.load_config()
    
    # Override config with command line arguments
    if args.dry_run:
        config_manager.system_config.dry_run = True
    
    config_manager.system_config.log_level = args.log_level
    
    # Create directories
    config_manager.create_directories()
    
    # Save current configuration
    config_manager.save_config()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(config_manager.training_config.logs_dir, 'main.log')
            )
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Print configuration summary
    config_manager.print_config_summary()
    
    # Set random seed
    set_seed(config_manager.system_config.seed)
    logger.info(f"Random seed set to {config_manager.system_config.seed}")
    
    # Setup device
    device = setup_device(config_manager)
    
    # Dry run warning
    if config_manager.system_config.dry_run:
        logger.warning("=" * 60)
        logger.warning("RUNNING IN DRY-RUN MODE")
        logger.warning("This will test the pipeline without actual training")
        logger.warning("Models will be minimal and no real checkpoints will be saved")
        logger.warning("=" * 60)
    
    try:
        # Initialize trainer
        logger.info("Initializing T5 distillation trainer...")
        trainer = T5DistillationTrainer(config_manager)
        
        # Setup training
        logger.info("Setting up training components...")
        trainer.setup_training()
        
        # Start training
        logger.info("Starting distillation training...")
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        # Cleanup
        if 'trainer' in locals():
            trainer.cleanup()
        logger.info("Program completed")

if __name__ == "__main__":
    main()