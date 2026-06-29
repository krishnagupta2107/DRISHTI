import os
from collections import Counter
import matplotlib.pyplot as plt
from src.data.dataset import MVSADataset

def run_eda(data_dir, label_file):
    """
    Runs Exploratory Data Analysis (EDA) on the parsed dataset and plots the class distribution.
    """
    dataset = MVSADataset(data_dir, label_file)
    print(f"Total Valid Samples: {len(dataset)}")
    
    labels = [s["label"] for s in dataset.samples]
    counts = Counter(labels)
    
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    print("\n--- Class Distribution ---")
    for k, v in sorted(counts.items()):
        print(f"{label_map[k]}: {v} samples")
        
    # Generate bar chart
    names = [label_map[k] for k in counts.keys()]
    values = list(counts.values())
    
    plt.figure(figsize=(8, 5))
    plt.bar(names, values, color=['#e74c3c', '#95a5a6', '#2ecc71'])
    plt.title('MVSA Dataset Sentiment Distribution')
    plt.xlabel('Sentiment Class')
    plt.ylabel('Number of Samples')
    
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/eda_distribution.png")
    print("\nEDA Plot saved successfully to logs/eda_distribution.png")

if __name__ == "__main__":
    # Hardcoded default paths for the EDA script execution
    run_eda(
        data_dir="c:/Users/Lenovo/OneDrive/Desktop/DRISHTI/MVSA/data", 
        label_file="c:/Users/Lenovo/OneDrive/Desktop/DRISHTI/MVSA/labelResultAll.txt"
    )
