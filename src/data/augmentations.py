import random
from torchvision import transforms

class ImageAugmentations:
    """Random transformations to prevent CNN overfitting."""
    
    @staticmethod
    def get_train_augmentations():
        """Full pipeline with Normalization for training."""
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_visual_augmentations():
        """Returns PIL images for Visualization (No ToTensor or Normalization)."""
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ])

class TextAugmentations:
    """Simple robust text augmentations for NLP robustness."""
    
    @staticmethod
    def random_swap(words, n=1):
        """Randomly swaps two words in a list of words."""
        if len(words) < 2:
            return words
        words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return words

    @staticmethod
    def random_delete(words, p=0.15):
        """Randomly deletes words with probability p."""
        if len(words) == 1:
            return words
        return [word for word in words if random.random() > p]

    @classmethod
    def apply_augmentation(cls, text, p_aug=0.3):
        """Applies a random text augmentation with probability p_aug."""
        if random.random() > p_aug:
            return text
            
        words = text.split()
        if len(words) < 2:
            return text
            
        aug_type = random.choice(["swap", "delete"])
        if aug_type == "swap":
            words = cls.random_swap(words)
        else:
            words = cls.random_delete(words)
            
        # Ensure we don't return an empty string if all words were deleted
        if not words:
            return text
            
        return " ".join(words)
