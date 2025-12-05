# Ocean Sound Classification with Deep Learning

A comprehensive deep learning project for classifying marine mammal sounds from the Watkins Marine Mammal Sound (WMMS) dataset. This project explores multiple neural network architectures and advanced techniques for audio classification using spectrograms.

## Project Overview

This project implements and compares **9 different deep learning approaches** for classifying 32 species of marine mammals from audio recordings:

**Tasks 1-5: Architecture Comparison**
- DNN on flattened spectrograms (38.97% accuracy)
- CNN on 2D spectrograms (57.35% accuracy)  Best traditional approach
- RNN for temporal modeling (12.50% accuracy)
- LSTM with attention mechanism (44.85% accuracy)
- Transformer with positional encoding (6.62% accuracy)

**Task 6: Raw Waveform Classification**
- 1D CNN on raw audio without spectrograms (63.24% accuracy)  Best overall

**Task 7: Contrastive Learning**
- Supervised contrastive loss for enhanced embeddings (27.57% accuracy)

**Task 8-9: Generative Data Augmentation**
- VAE-based augmentation (68.75% accuracy)  Best with augmentation
- Diffusion model augmentation (44.49% accuracy)

## Results

| Method | Test Accuracy | Improvement |
|--------|--------------|-------------|
| DNN (Baseline) | 38.97% | - |
| CNN | 57.35% | +18.38% |
| Raw Waveform CNN | 63.24% | +24.27% |
| **CNN + VAE Augmentation** | **68.75%** | **+29.78%** |

- **CNNs outperform sequential models** (RNN/LSTM/Transformer) for spectrogram classification, as they preserve 2D spatial structure
- **Raw waveform learning** is competitive with spectrograms, enabling end-to-end feature learning
- **VAE augmentation** significantly improves performance (+11.40%) by generating realistic synthetic samples
- **Transformers fail** on small datasets without sufficient training data or appropriate inductive biases
- **Contrastive learning underperforms** traditional supervised learning on limited data

## Technologies

- **PyTorch** for deep learning models
- **Scipy** for spectrogram generation
- **scikit-learn** for evaluation metrics
- **NumPy/Pandas** for data processing
- **Matplotlib/Seaborn** for visualizations

## Dataset

Watkins Marine Mammal Sound (WMMS) dataset containing audio recordings of 32 marine mammal species including dolphins, whales, and seals. Visit my kaggle account for dataset https://www.kaggle.com/datasets/depakkaggle/oceansound. 

## Models Implemented

- Fully Connected DNN
- 2D CNN with batch normalization
- Bidirectional LSTM with attention
- Transformer with CLS token
- 1D CNN for raw waveforms
- Contrastive learning encoder
- Variational Autoencoder (VAE)
- Conditional Diffusion Model (DDPM)

*This project demonstrates comprehensive exploration of deep learning techniques for audio classification, from basic architectures to advanced generative models.*
