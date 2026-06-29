import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from src.data.augmentations import ImageAugmentations

def run_augmentation_visualization(data_dir):
    """
    Randomly picks an image, applies augmentations 3 times, and saves a plot.
    """
    # Find all jpgs in the directory
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    if not all_files:
        print(f"Error: No images found in {data_dir}")
        return
        
    random_img_name = random.choice(all_files)
    img_path = os.path.join(data_dir, random_img_name)
    
    print(f"Visualizing augmentations for: {random_img_name}")
    original = Image.open(img_path).convert("RGB")
    
    # Get the visualization transforms (No Tensors)
    aug_transform = ImageAugmentations.get_visual_augmentations()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    for i in range(1, 4):
        augmented = aug_transform(original)
        axes[i].imshow(augmented)
        axes[i].set_title(f"Augmented Version {i}")
        axes[i].axis('off')
        
    os.makedirs("logs", exist_ok=True)
    save_path = "logs/augmentation_demo.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    run_augmentation_visualization("c:/Users/Lenovo/OneDrive/Desktop/DRISHTI/MVSA/data")
