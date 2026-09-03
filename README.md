# 🏠 Airbnb Price Prediction System

An end-to-end Machine Learning application that predicts the estimated nightly price of Airbnb listings in New York City.

The project combines data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and a user-friendly Streamlit web application.

## 🚀 Live Demo

👉 [Try the Airbnb Price Prediction System](https://airbnb-price-prediction-system.streamlit.app/)

---

## 📌 Project Overview

The goal of this project is to estimate the nightly price of an Airbnb listing based on its location, property characteristics, reviews, rating, availability, and other listing information.

The complete workflow includes:

- Data cleaning and preprocessing
- Missing-value handling
- Exploratory Data Analysis (EDA)
- Outlier analysis
- Feature engineering
- Location-based feature creation
- Machine Learning model training
- Model comparison and evaluation
- Model serialization using Joblib
- Streamlit deployment

---

## ✨ Features

### 🏡 Property Information
- Borough
- Neighbourhood
- Room type
- Number of bedrooms
- Number of beds
- Number of bathrooms

### 📊 Listing Information
- Minimum nights
- Number of reviews
- Reviews per month
- Host listing count
- Availability
- Guest rating

### 📍 Location Features

The model also uses location-based features derived from latitude and longitude, including approximate distances from:

- Midtown Manhattan
- Lower Manhattan
- Central Park

These features help the model capture the relationship between location and Airbnb pricing.

---

## 🤖 Machine Learning

Several regression models were evaluated during the project:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

The final model uses **XGBoost Regression** with preprocessing and feature engineering integrated into a Scikit-learn Pipeline.

### Final Model

**XGBoost Regressor**

The model was trained on listings with observed prices up to **$500 per night** to reduce the influence of extreme high-price observations.

### Model Performance

| Metric | Result |
|---|---:|
| MAE | **$41.46** |
| RMSE | **$59.97** |
| R² Score | **0.5654** |

The final model achieved an R² score of approximately **0.57**, improving upon the earlier Random Forest model with an R² of approximately **0.36**.

> Note: The model is designed for price estimation within the range represented by the training data. Predictions outside the training price range should be interpreted cautiously.

---

## 🛠️ Tech Stack

### Programming
- Python

### Machine Learning
- Scikit-learn
- XGBoost

### Data Analysis
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Application
- Streamlit

### Model Persistence
- Joblib

### Development Environment
- Jupyter Notebook

### Deployment
- Streamlit Community Cloud

---

## 📂 Project Structure

```text
Airbnb_Price/
│
├── app/
│   └── app.py
│
├── models/
│   └── airbnb_price_model.pkl
│
├── notebooks/
│   └── Airbnb_Price_Prediction.ipynb
│
├── data/
│   └── new_york_listings_2024.csv
│
├── requirements.txt
├── README.md
└── .gitignore