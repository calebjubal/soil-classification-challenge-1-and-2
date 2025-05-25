# Annam.ai Internship - Soil Classification Challenges

This repository contains Python solutions for two soil classification challenges completed during an internship at Annam.ai. The code is implemented using PyTorch and leverages Jupyter Notebooks for both training and inference.

## Team Information

*   **Author**: Annam.ai IIT Ropar
*   **Team Name**: SoilClassifiers
*   **Team Members**: Caleb Chandrasekar, Sarvesh Chandran, Swaraj Bhattacharjee, Karan Singh, Saatvik Tyagi
*   **Leaderboard Ranks**:
    *   Challenge 1: 103
    *   Challenge 2: 120

## About the Challenges & Code

### Challenge 1: Soil Type Classification
*   **Goal**: Classify soil images into four types (Alluvial, Black, Clay, Red).
*   **Approach**: Fine-tuned an `efficientnet_v2_s` model. Inference utilizes Test-Time Augmentation (TTA).
*   **Code**:
    *   `challenge-1/training.ipynb`: Trains the classification model.
    *   `challenge-1/inference.ipynb`: Performs predictions with TTA.

### Challenge 2: Autoencoder for Anomaly Detection
*   **Goal**: Detect anomalous soil images using an autoencoder.
*   **Approach**: Trained a convolutional autoencoder. Images are classified based on reconstruction error against a dynamically calculated threshold.
*   **Code**:
    *   `challenge-2/training.ipynb`: Trains the autoencoder and determines the anomaly threshold.
    *   `challenge-2/inference.ipynb`: Uses the trained autoencoder to classify test images.

## Key Technologies
*   Python
*   PyTorch
*   Torchvision
*   Pandas
*   NumPy
*   Scikit-learn
*   Pillow (PIL)

For detailed implementation, model architectures, and parameters, please refer to the respective Jupyter Notebooks within the `challenge-1` and `challenge-2` directories.