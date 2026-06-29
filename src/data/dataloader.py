import torch
from torch.utils.data import DataLoader, random_split
from src.data.dataset import MVSADataset
import logging

logger = logging.getLogger("DrishtiLogger")

def get_dataloaders(data_dir, label_file, batch_size=4, num_workers=0):
    """
    Builds Training, Validation, and Testing DataLoaders with an 80/10/10 split.
    """
    dataset = MVSADataset(data_dir, label_file)
    
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    
    # Deterministic split for reproducibility
    train_ds, val_ds, test_ds = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Dataset Split -> Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Custom collate function to handle un-transformed PIL images and raw text strings
    def collate_fn(batch):
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        ids = [b["id"] for b in batch]
        
        return {
            "images": images, 
            "texts": texts, 
            "labels": labels, 
            "ids": ids
        }
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader
