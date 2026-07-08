# 🌱 AI-Powered Crop Health Prediction System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📖 Overview

Crop diseases significantly reduce agricultural productivity worldwide. This project combines **Machine Learning** and **Deep Learning** to build an intelligent crop health prediction system.

The application provides **two prediction methods**:

- 📊 **Feature-Based Prediction** using an **XGBoost Classifier**
- 📷 **Image-Based Disease Detection** using a **CNN (TensorFlow/Keras)**

The web application is developed using **Streamlit**, allowing users to analyze crop health through agricultural parameters or by uploading crop leaf images.

---

# ✨ Features

### 📊 Feature Prediction

Predict crop health using agricultural parameters such as:

- Temperature
- Humidity
- Rainfall
- Soil Moisture
- Soil pH
- NDVI
- SAVI
- Chlorophyll Content
- Leaf Area Index
- Crop Growth Stage
- Crop Type
- Pest Damage
- Weed Coverage
- Organic Matter
- Canopy Coverage
- and many more...

**Model Used**

- ✅ XGBoost Classifier

---

### 📷 Image Prediction

Upload a crop leaf image and detect disease automatically.

Supported Classes:

- 🌿 Healthy
- 🍂 Rust
- 🍃 Powdery

**Model Used**

- ✅ Convolutional Neural Network (CNN)
- TensorFlow / Keras

---

# 🧠 Machine Learning Pipeline

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Label Encoding
- Train/Test Split
- Model Training
- Hyperparameter Tuning
- Performance Evaluation
- Streamlit Deployment

---

# 🏗 Deep Learning Pipeline

- Image Augmentation
- Image Resizing (224×224)
- CNN Architecture
- Model Training
- Validation
- Testing
- Saved as `.keras`

---

# 📂 Datasets

## 📊 Agricultural Dataset

Used for Feature Prediction

https://www.kaggle.com/code/valentinohartanto/crop-health-eda/input?select=agriculture_dataset.csv

Includes features like

- NDVI
- SAVI
- Soil Moisture
- Temperature
- Humidity
- Rainfall
- Crop Stress Indicator
- Crop Growth Stage
- Soil pH
- Organic Matter
- Crop Health Label

---

## 📷 Plant Disease Image Dataset

Used for CNN Model

https://www.kaggle.com/code/vad13irt/plant-disease-classification/input

Classes:

- Healthy
- Rust
- Powdery

---

# 🤖 Models Used

| Model | Purpose |
|---------|---------|
| XGBoost Classifier | Feature Prediction |
| CNN (TensorFlow/Keras) | Image Disease Detection |

---

# 🛠 Technologies Used

- Python
- Streamlit
- TensorFlow
- Keras
- XGBoost
- Scikit-Learn
- NumPy
- Pandas
- Pillow
- Joblib
- Matplotlib
- Plotly

---

# 📁 Project Structure

```
Crop-Health-Prediction/
│
├── app.py
├── agriculture.ipynb
├── crop_diseases.ipynb
├── model.pkl
├── crop_disease_cnn.keras
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📈 Model Output

## Feature Prediction

Input agricultural parameters

↓

XGBoost Classifier

↓

Healthy / Disease

↓

Confidence Score

---

## Image Prediction

Upload Leaf Image

↓

CNN Model

↓

Healthy / Rust / Powdery

↓

Prediction Confidence

---

# 🎯 Applications

- Smart Farming
- Precision Agriculture
- Disease Monitoring
- Crop Health Analysis
- Educational Purpose
- Agricultural Research

---

# 🔮 Future Improvements

- 🌦 Live Weather API
- 📍 GPS Integration
- 🛰 Satellite Image Analysis
- 🌱 Fertilizer Recommendation
- 💧 Irrigation Recommendation
- 📄 PDF Report Generation
- 🔊 Voice Assistant
- 🌐 Multi-language Support
- ☁ Cloud Deployment
- 📱 Android Application

---

# 📊 Libraries

```
streamlit
tensorflow
keras
xgboost
scikit-learn
numpy
pandas
joblib
Pillow
matplotlib
plotly
```

---

# 👨‍💻 Author

**Mohit Gaur**

Computer Science Student

Machine Learning | Deep Learning | AI | Data Science

GitHub:
https://github.com/Mohit260805

---

# ⭐ If you found this project useful

Please consider giving this repository a ⭐ on GitHub.

It motivates me to build more AI projects.

---

## 📜 License

This project is licensed under the **MIT License**.
