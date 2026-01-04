# EmotiSense 🎭  
### Real-Time Facial Emotion Detection System

EmotiSense is a real-time emotion detection system that analyzes facial expressions from a live camera feed and predicts human emotions using computer vision and machine learning techniques. The project focuses on efficiency, interpretability, and real-time performance using classical machine learning models instead of heavy deep learning frameworks.

---

## 📌 Overview

EmotiSense captures live video input, detects human faces, preprocesses facial features, and classifies emotions in real time. The system is trained on labeled facial images and supports checkpoint-based training, class imbalance handling, and model persistence.

This project demonstrates a complete end-to-end machine learning pipeline — from data loading and preprocessing to training, evaluation, and deployment-ready model saving.

---

## ✨ Key Features

- 🎥 Real-time facial emotion detection
- 🙂 Supports multiple emotions (Happy, Sad, Angry, Neutral, Surprise, Fear, etc.)
- 🧠 Machine learning–based classification using SGDClassifier
- ⚖️ Handles class imbalance using class weights
- 🔁 Checkpoint-based training with resume support
- 💾 Best model auto-saved based on validation accuracy
- ⚡ Lightweight and fast (no deep learning frameworks required)

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV** – face detection & image processing
- **NumPy** – numerical operations
- **scikit-learn**
  - SGDClassifier
  - StandardScaler
  - Train–validation split
  - Accuracy evaluation
- **Joblib** – model & checkpoint persistence

---

## 📂 Project Structure

