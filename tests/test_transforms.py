import torch
from PIL import Image
from src.data.transforms import ImageTransforms, TextProcessor

def test_image_transforms():
    """Ensure dummy image is properly resized and converted to Tensor."""
    img = Image.new("RGB", (500, 500))
    transform = ImageTransforms.get_val_transforms()
    tensor_img = transform(img)
    
    assert isinstance(tensor_img, torch.Tensor), "Output should be a PyTorch Tensor"
    assert tensor_img.shape == (3, 224, 224), "Image shape must strictly be 3x224x224"

def test_text_processor():
    """Ensure tokenization creates correct shapes and langdetect works."""
    processor = TextProcessor(model_name="xlm-roberta-base", max_length=32)
    
    # Test Language Detection
    text_en = "Hello world, this is a test!"
    text_hi = "नमस्ते दुनिया"
    
    assert processor.detect_lang(text_en) == "en", "Failed to detect English"
    assert processor.detect_lang(text_hi) in ["hi", "mr", "ne"], "Failed to detect Hindi/Marathi script"
    
    # Test Tokenization Shape
    tokens = processor.tokenize(text_en)
    assert "input_ids" in tokens, "Missing input_ids"
    assert "attention_mask" in tokens, "Missing attention_mask"
    
    # Shape should be [1, max_length] because we pass a single string
    assert tokens["input_ids"].shape == (1, 32), "Token shape mismatch"
