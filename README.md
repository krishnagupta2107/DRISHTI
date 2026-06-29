# DRISHTI: Cross-Lingual Multimodal Sentiment Analysis

DRISHTI is a research-grade Deep Learning system designed to perform sentiment analysis by jointly understanding both **images** and **text** across 7 different languages (English, Hindi, Bengali, Tamil, Telugu, Marathi, Urdu).

## Overview

In traditional NLP, sentiment analysis is performed solely on text. However, in the real world (e.g., social media), a positive image accompanied by a sarcastic text caption might actually convey a negative sentiment. DRISHTI bridges this gap by fusing Computer Vision and Natural Language Processing.

### Core Architecture
- **Vision Encoder:** ResNet50 / ViT-B/16 (Extracts spatial features from images)
- **Text Encoder:** XLM-RoBERTa (Extracts cross-lingual semantic features from text)
- **Fusion Mechanism:** Cross-Attention Gating (Mathematically aligns and fuses image and text vectors)
- **Interpretability:** Grad-CAM & SHAP (Visualizes *why* the model made a specific prediction)

## Tech Stack
- **Deep Learning Engine:** PyTorch, Torchvision, Transformers (HuggingFace)
- **Experiment Tracking:** Weights & Biases (W&B)
- **Deployment:** FastAPI, React, Docker (Upcoming)
