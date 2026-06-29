from src.data.augmentations import TextAugmentations

def test_text_augmentations():
    """Validates the text robustness algorithms."""
    text = "The quick brown fox jumps over the lazy dog"
    words = text.split()
    
    # 1. Test Random Swap
    swapped = TextAugmentations.random_swap(words, n=1)
    assert len(swapped) == len(words), "Swapping should not change length"
    assert swapped != words, "Words should be swapped"
    
    # 2. Test Random Delete (force 100% deletion for test)
    deleted = TextAugmentations.random_delete(words, p=1.0)
    assert len(deleted) == 0, "All words should have been deleted"
    
    # 3. Test Edge Case: Single word sentence
    single_word = ["Hello"]
    swapped_single = TextAugmentations.random_swap(single_word)
    assert swapped_single == single_word, "Single word cannot be swapped"
    
    deleted_single = TextAugmentations.random_delete(single_word, p=1.0)
    assert deleted_single == single_word, "Single word cannot be deleted (to prevent empty strings)"
    
    # 4. Test High-Level Apply Function
    aug_text = TextAugmentations.apply_augmentation(text, p_aug=1.0)
    assert isinstance(aug_text, str), "Must return a string"
