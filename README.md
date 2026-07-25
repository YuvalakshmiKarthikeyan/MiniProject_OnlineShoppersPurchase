# 🛒 Online Shoppers Purchasing Intention Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Project Overview

This project predicts whether an online shopper will complete a purchase based on their browsing behavior. Multiple machine learning classification algorithms were developed, evaluated, and compared. After hyperparameter tuning, the **XGBoost classifier** was selected as the final model due to its high predictive performance. The trained model was deployed using **FastAPI** and containerized with **Docker** to provide real-time purchase predictions.

---

# 📚 Table of Contents

- Project Overview
- Problem Statement
- Objectives
- Project Type
- Dataset Information
- Technology Stack
- Project Workflow
- Model Performance Comparison
- Hyperparameter Tuning
- Final Model
- Deployment
- Project Structure
- Installation
- API Usage
- Results
- Future Enhancements
- Author

---

# 🎯 Problem Statement

Online retailers receive thousands of website visitors every day, but only a small percentage complete a purchase. Predicting purchase intention enables businesses to identify potential buyers, improve marketing strategies, personalize recommendations, and increase conversion rates.

---

# 📌 Project Type

**Supervised Machine Learning**

✔ Classification

---

# 🎯 Objectives

- Analyze online shopper browsing behavior.
- Perform data preprocessing and feature engineering.
- Compare multiple classification algorithms.
- Handle class imbalance using SMOTE.
- Optimize the best-performing model through hyperparameter tuning.
- Deploy the trained model using FastAPI and Docker.

---

# 📊 Dataset Information

### Dataset Source

UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

### Dataset Description

The dataset contains customer browsing session information collected from an e-commerce website. Features include administrative pages visited, product pages viewed, browsing duration, visitor type, page values, bounce rates, and purchase outcome.

### Dataset Summary

| Attribute | Value |
|------------|-------|
| Records | 12,330 |
| Features | 17 |
| Target Variable | Revenue |
| Problem Type | Binary Classification |

---

# 💻 Technology Stack

| Category | Technologies |
|------------|----------------|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Handling Imbalanced Data | SMOTE |
| API Framework | FastAPI |
| Deployment | Docker |
| IDE | Jupyter Notebook, VS Code |

---

# 🔄 Project Workflow

```
Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Feature Selection
      │
      ▼
Train-Test Split
      │
      ▼
Model Training (Before SMOTE)
      │
      ▼
SMOTE
      │
      ▼
Model Training (After SMOTE)
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Final Tuned XGBoost Model
      │
      ▼
FastAPI Deployment
      │
      ▼
Docker Deployment
```

---

# 📊 Exploratory Data Analysis

Performed

- Dataset Overview
- Missing Value Analysis
- Duplicate Removal
- Outlier Detection
- Correlation Analysis
- Distribution Analysis
- Data Visualization

### Key Insights

- No missing values were found.
- 125 duplicate records were removed.
- BounceRates and ExitRates were highly correlated.
- ExitRates was removed to reduce multicollinearity.
- PageValues showed the strongest correlation with Revenue.
- ProductRelated and ProductRelated_Duration showed strong positive correlation.

---

# ⚙ Data Preprocessing

Performed

- Missing Value Check
- Duplicate Removal
- Outlier Treatment using IQR
- Label Encoding
- Feature Scaling using StandardScaler
- Train-Test Split (80:20)

---

# 🧩 Feature Engineering & Selection

### Feature Engineering

- Total_Page_Visits
- Total_Duration

### Feature Selection

- Correlation Analysis
- SelectKBest

### Final Features

- Administrative
- Informational
- ProductRelated
- ProductRelated_Duration
- BounceRates
- PageValues
- VisitorType
- Total_Duration
- Total_Page_Visits

---

# 🤖 Model Building

## 📌 Before SMOTE

### Class Distribution

| Revenue | Count |
|---------|------:|
| No Purchase | 8238 |
| Purchase | 1526 |

Models

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

---

## ⚖ After SMOTE

### Class Distribution

| Revenue | Count |
|---------|------:|
| No Purchase | 8238 |
| Purchase | 8238 |

Models

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

---

# 📈 Model Performance Comparison

| Model | Accuracy (Before) | Accuracy (After) | Recall (Class 1) | F1-Score (Class 1) | True Positives | False Negatives | Overall Analysis |
|------|:-----------------:|:----------------:|:----------------:|:------------------:|:--------------:|:---------------:|-----------------|
| Logistic Regression | **88.69%** | **87.30%** | 0.39 → **0.76** | 0.52 → **0.65** | 149 → **290** | 233 → **92** | Better minority class detection after SMOTE. |
| Decision Tree | **87.10%** | **85.13%** | 0.60 → **0.63** | 0.59 → 0.57 | 229 → **240** | 153 → **142** | Slight improvement in recall with minor accuracy reduction. |
| Random Forest | **90.70%** | **88.57%** | 0.61 → **0.74** | 0.67 → 0.67 | 232 → **282** | 150 → **100** | Strong balanced performance after SMOTE. |
| Support Vector Machine | **89.06%** | **86.64%** | 0.46 → **0.72** | 0.57 → **0.63** | 177 → **276** | 205 → **106** | Significant improvement in minority class prediction. |
| XGBoost | **90.21%** | **89.76%** | 0.61 → **0.69** | 0.66 → **0.68** | 234 → **264** | 148 → **118** | Best balance between accuracy and recall. |

> **Observation:** Applying **SMOTE** significantly improved recall and reduced false negatives for all models, making them more effective at identifying purchasing customers.

---

# 🎯 Hyperparameter Tuning

RandomizedSearchCV was used to optimize:

- Random Forest
- XGBoost

---

# 🏆 Final Model

## Tuned XGBoost

### Performance

| Metric | Value |
|---------|-------|
| Accuracy | **88.53%** |
| ROC-AUC Score | **0.9334** |
| Training Accuracy | **92.56%** |
| Testing Accuracy | **88.53%** |

### Why XGBoost?

- Highest predictive performance
- Excellent ROC-AUC Score
- Balanced Precision and Recall
- Reduced overfitting
- Better generalization

---

# 🚀 Deployment

The final XGBoost model was deployed using **FastAPI**.

Docker was used to containerize the application for consistent deployment across different environments.

### API Features

- Load trained model
- Load scaler
- Accept user input
- Predict purchase intention
- Return purchase probability

Swagger UI

```
http://localhost:8000/docs
```

---

# 📁 Project Structure

```
Online_Shoppers_Purchasing_Intention/
│
├── dataset/
│   └── online_shoppers_intention.csv
│
├── notebooks/
│   └── Main_Project.ipynb
│
├── models/
│   ├── best_xgb_model.pkl
│   └── scaler.pkl
│
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
```

---

# ⚙ Installation

Clone Repository

```bash
git clone https://github.com/yourusername/Online-Shoppers-Purchasing-Intention.git
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run FastAPI

```bash
uvicorn app:app --reload
```

---

## Docker

Build Image

```bash
docker build -t online-shopper-api .
```

Run Container

```bash
docker run -p 8000:8000 online-shopper-api
```

---

# 🌐 API Endpoint

| Method | Endpoint | Description |
|----------|-----------|-------------|
| POST | /predict | Predict customer purchase intention |

---

# 📌 Results

- Successfully compared multiple classification algorithms.
- Addressed class imbalance using SMOTE.
- Optimized XGBoost using RandomizedSearchCV.
- Achieved **88.53% accuracy**.
- Achieved **0.9334 ROC-AUC Score**.
- Successfully deployed using FastAPI.
- Containerized the application using Docker.

---

# 🔮 Future Enhancements

- Integrate with a live e-commerce application.
- Deploy on AWS, Azure, or Google Cloud.
- Implement automated model retraining.
- Develop an interactive analytics dashboard.
- Add authentication and API security.
- Monitor model performance in production.

---

# 👩‍💻 Author

**Yuvalakshmi Karthikeyan**

Machine Learning Engineer | Data Science Enthusiast

---

## ⭐ If you found this project useful, consider giving it a Star!
