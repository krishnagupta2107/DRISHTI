import os
from collections import Counter
import logging
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger("DrishtiLogger")

class MVSADataset(Dataset):
    """
    PyTorch Dataset for the Multi-View Sentiment Analysis (MVSA) dataset.
    Reads image-text pairs and resolves multi-annotator labels via majority vote.
    """
    def __init__(self, data_dir, label_file, image_transform=None, text_transform=None):
        self.data_dir = data_dir
        self.image_transform = image_transform
        self.text_transform = text_transform
        
        self.samples = self._parse_labels(label_file)
        logger.info(f"Loaded {len(self.samples)} valid multimodal samples from {label_file}")

    def _parse_labels(self, label_file):
        samples = []
        label_map = {"positive": 2, "neutral": 1, "negative": 0}
        
        # Read file, skipping header
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]
            
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
                
            img_id = parts[0]
            
            # Flatten all 6 annotations (3 text, 3 image from the 3 annotators)
            annotations = []
            for p in parts[1:]:
                txt_lbl, img_lbl = p.split(',')
                annotations.extend([txt_lbl, img_lbl])
                
            # Compute Majority Vote
            counter = Counter(annotations)
            majority_label = counter.most_common(1)[0][0]
            
            if majority_label not in label_map:
                continue
                
            img_path = os.path.join(self.data_dir, f"{img_id}.jpg")
            txt_path = os.path.join(self.data_dir, f"{img_id}.txt")
            
            # Validate that both modalities exist on disk
            if os.path.exists(img_path) and os.path.exists(txt_path):
                samples.append({
                    "id": img_id,
                    "image_path": img_path,
                    "text_path": txt_path,
                    "label": label_map[majority_label]
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Safely load Image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {sample['image_path']}: {e}")
            image = Image.new("RGB", (224, 224)) # Dummy blank image fallback
            
        # Safely load Text
        try:
            with open(sample["text_path"], 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
        except Exception as e:
            logger.error(f"Error loading text {sample['text_path']}: {e}")
            text = ""
            
        # Apply preprocessing transforms (To be injected in Milestone 3)
        if self.image_transform:
            image = self.image_transform(image)
        if self.text_transform:
            text = self.text_transform(text)
            
        return {
            "image": image,
            "text": text,
            "label": sample["label"],
            "id": sample["id"]
        }
