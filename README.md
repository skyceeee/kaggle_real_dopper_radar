# Doppler Radar CNN Classifier 🚗✈️🧍

This project focuses on **classifying cars, drones, and people** using **Doppler radar signals** with deep learning.  
Dataset: [Real Doppler Radar Database (Kaggle)](https://www.kaggle.com/datasets/iroldan/real-doppler-raddar-database)

## 🚀 Overview
- **RadarCSVSet Dataset Class** → loads radar `.csv` files, normalizes, and applies augmentations.  
- **RadarCNN** → custom CNN model built with PyTorch (conv + pooling + dropout).  
- **Optuna Hyperparameter Optimization** → tunes learning rate, batch size, weight decay, dropout, etc.  
- **5-Fold Cross Validation** → provides more reliable performance estimation.  
- **Ensemble Inference** → averages predictions from 5 folds with softmax, improving stability and accuracy.  
- **Evaluation Report** → Precision, Recall, F1-score, Confusion Matrix.  

## 📊 Results
- **Best single fold** macro-F1 ≈ **0.97**  
- **5-Fold Ensemble** macro-F1 ≈ **0.98** ✅  

## 🛠 Tech Stack
- Python 3.11  
- PyTorch  
- Optuna  
- scikit-learn  
- NumPy, Pandas, Matplotlib  
- Kaggle Notebooks  

