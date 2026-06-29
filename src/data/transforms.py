from torchvision import transforms
from transformers import AutoTokenizer
from langdetect import detect, DetectorFactory
import logging

logger = logging.getLogger("DrishtiLogger")
# Enforce deterministic results for language detection
DetectorFactory.seed = 42

class ImageTransforms:
    """Standard ImageNet transforms for ResNet50 and ViT-B/16."""
    
    @staticmethod
    def get_train_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Normalization based on ImageNet statistics
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_val_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class TextProcessor:
    """Handles Tokenization and Language Detection."""
    
    def __init__(self, model_name="xlm-roberta-base", max_length=128):
        logger.info(f"Initializing Tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def tokenize(self, text: str):
        """
        Converts raw text into input_ids and attention_masks.
        Outputs PyTorch tensors ('pt').
        """
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
    def detect_lang(self, text: str) -> str:
        """
        Detects the language of the text. Returns ISO language code (e.g., 'en', 'hi', 'mr').
        """
        try:
            if len(text.strip()) > 0:
                return detect(text)
            return "unknown"
        except Exception:
            return "unknown"
