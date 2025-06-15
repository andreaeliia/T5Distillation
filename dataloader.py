"""
Data loading and preprocessing module for T5 distillation
"""
import logging
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import T5Tokenizer
from config import DataConfig, SystemConfig

class CNNDailyMailDataset(Dataset):
    """Dataset class for CNN/DailyMail summarization task"""
    
    def __init__(
        self,
        data: HFDataset,
        tokenizer: T5Tokenizer,
        max_input_length: int,
        max_target_length: int,
        dry_run: bool = False
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.dry_run = dry_run
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Dataset initialized with {len(self.data)} samples")
        
        if self.dry_run:
            self.logger.info("DRY RUN MODE: Using minimal dummy data")
    
    def __len__(self) -> int:
        if self.dry_run:
            return min(10, len(self.data))  # Use only 10 samples in dry run
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.dry_run and idx >= 10:
            idx = idx % 10  # Cycle through first 10 samples
            
        item = self.data[idx]
        
        # Prepare input text with T5 prefix
        input_text = f"summarize: {item['article']}"
        target_text = item['highlights']
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels (replace padding token id with -100)
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'decoder_input_ids': target_encoding['input_ids'].squeeze(),
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze()
        }

class DataLoaderManager:
    """Manager for data loading operations"""
    
    def __init__(self, data_config: DataConfig, system_config: SystemConfig):
        self.data_config = data_config
        self.system_config = system_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize tokenizer
        self.tokenizer = None
        self._datasets = {}
        
    def setup_tokenizer(self, model_name: str) -> T5Tokenizer:
        """Setup T5 tokenizer"""
        self.logger.info(f"Loading tokenizer for {model_name}")
        
        if self.system_config.dry_run:
            self.logger.info("DRY RUN MODE: Loading minimal tokenizer")
            
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            cache_dir=self.data_config.cache_dir
        )
        
        self.logger.info("Tokenizer loaded successfully")
        return self.tokenizer
    
    def load_dataset(self) -> Tuple[HFDataset, HFDataset, HFDataset]:
        """Load CNN/DailyMail dataset"""
        self.logger.info(f"Loading {self.data_config.dataset_name} dataset")
        
        if self.system_config.dry_run:
            self.logger.info("DRY RUN MODE: Loading minimal dataset samples")
        
        try:
            # Load dataset
            dataset = load_dataset(
                self.data_config.dataset_name,
                self.data_config.dataset_version,
                cache_dir=self.data_config.cache_dir
            )
            
            # Get train, validation, test splits
            train_dataset = dataset['train']
            val_dataset = dataset['validation']
            test_dataset = dataset['test']
            
            # Limit dataset size if specified or in dry run mode
            if self.system_config.dry_run:
                train_size = min(50, len(train_dataset))
                val_size = min(20, len(val_dataset))
                test_size = min(10, len(test_dataset))
            else:
                train_size = self.data_config.train_size or len(train_dataset)
                val_size = self.data_config.val_size or len(val_dataset)
                test_size = self.data_config.test_size or len(test_dataset)
            
            if train_size < len(train_dataset):
                train_dataset = train_dataset.select(range(train_size))
            if val_size < len(val_dataset):
                val_dataset = val_dataset.select(range(val_size))
            if test_size < len(test_dataset):
                test_dataset = test_dataset.select(range(test_size))
            
            self.logger.info(f"Dataset loaded - Train: {len(train_dataset)}, "
                           f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def create_datasets(
        self,
        train_data: HFDataset,
        val_data: HFDataset,
        test_data: HFDataset,
        max_input_length: int,
        max_target_length: int
    ) -> Tuple[CNNDailyMailDataset, CNNDailyMailDataset, CNNDailyMailDataset]:
        """Create PyTorch datasets"""
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup_tokenizer first.")
        
        self.logger.info("Creating PyTorch datasets")
        
        train_dataset = CNNDailyMailDataset(
            train_data, self.tokenizer, max_input_length, 
            max_target_length, self.system_config.dry_run
        )
        
        val_dataset = CNNDailyMailDataset(
            val_data, self.tokenizer, max_input_length, 
            max_target_length, self.system_config.dry_run
        )
        
        test_dataset = CNNDailyMailDataset(
            test_data, self.tokenizer, max_input_length, 
            max_target_length, self.system_config.dry_run
        )
        
        self.logger.info("PyTorch datasets created successfully")
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(
        self,
        train_dataset: CNNDailyMailDataset,
        val_dataset: CNNDailyMailDataset,
        test_dataset: CNNDailyMailDataset,
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders"""
        
        self.logger.info(f"Creating data loaders with batch size: {batch_size}")
        
        # Adjust batch size for dry run
        if self.system_config.dry_run:
            batch_size = min(2, batch_size)
            self.logger.info(f"DRY RUN MODE: Using batch size {batch_size}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers if not self.system_config.dry_run else 0,
            pin_memory=True if self.system_config.device == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers if not self.system_config.dry_run else 0,
            pin_memory=True if self.system_config.device == "cuda" else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers if not self.system_config.dry_run else 0,
            pin_memory=True if self.system_config.device == "cuda" else False
        )
        
        self.logger.info("Data loaders created successfully")
        self.logger.info(f"Train batches: {len(train_loader)}, "
                        f"Val batches: {len(val_loader)}, "
                        f"Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def get_sample_batch(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """Get a sample batch for testing"""
        self.logger.info("Getting sample batch for inspection")
        
        for batch in dataloader:
            self.logger.info(f"Sample batch shapes:")
            for key, tensor in batch.items():
                self.logger.info(f"  {key}: {tensor.shape}")
            return batch
        
        raise ValueError("DataLoader is empty")