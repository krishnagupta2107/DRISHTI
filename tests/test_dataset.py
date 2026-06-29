import os
from src.data.dataset import MVSADataset
from src.data.dataloader import get_dataloaders

def test_dataset_loading():
    """
    Validates that the MVSA dataset parses correctly and Dataloaders yield proper batches.
    """
    data_dir = "MVSA/data"
    label_file = "MVSA/labelResultAll.txt"
    
    # Only run the test if the user actually has the data downloaded
    if os.path.exists(data_dir) and os.path.exists(label_file):
        dataset = MVSADataset(data_dir, label_file)
        
        # 1. Test Dataset Length
        assert len(dataset) > 0, "Dataset parsed 0 samples. Check label file format."
        
        # 2. Test single sample retrieval
        sample = dataset[0]
        assert "image" in sample, "Image missing from sample dict"
        assert "text" in sample, "Text missing from sample dict"
        assert "label" in sample, "Label missing from sample dict"
        assert isinstance(sample["text"], str), "Text should be loaded as a string"
        
        # 3. Test Dataloader batching (Collate function test)
        train_loader, _, _ = get_dataloaders(data_dir, label_file, batch_size=2)
        batch = next(iter(train_loader))
        
        assert len(batch["images"]) == 2, "Batch should contain exactly 2 images"
        assert len(batch["texts"]) == 2, "Batch should contain exactly 2 texts"
        assert batch["labels"].shape == (2,), "Batch labels should have shape (2,)"
