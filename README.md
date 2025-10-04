# 🎓 Student Dropout Risk Prediction App

This project is a **Streamlit web application** that predicts the risk of student dropout based on academic and demographic data. It uses **machine learning models** trained on student performance to help advisors, teachers, and administrators identify at-risk students and intervene early.

The app consists of two prediction modules:

  • **After 1st Semester** → Predicts dropout risk using student features and 1st semester performance.
  
  • **After 2nd Semester** → Predicts dropout risk using features and performance after 2nd semester, for more complete assessment.

Both models include **explainable AI (XAI)** using **SHAP**, so you can see which features most influenced the prediction.

## 🚀 Features

  • Predict dropout probability for students **after 1st semester** or **after 2nd semester**.
  
  • **Explainable AI (SHAP)** to show the top features affecting the model’s decision.
  
  • **Interactive Streamlit interface** with easy-to-use input fields.
  
  • Modular design for future extension with more models or datasets.


## ▶️ Usage

Run the app locally:

streamlit run app.py


## 📊 Models

The app uses a **VotingClassifier** ensemble with:

  • Logistic Regression
  
  • Random Forest Classifier
  
  • AdaBoost Classifier
  
  • XGBoost Classifier

Scaling is applied with **RobustScaler**. Models were trained separately for **after 1st semester** and **after 2nd semester** scenarios.

## 🤖 Explainable AI (XAI)

We use **SHAP (SHapley Additive exPlanations)** to identify the most important features influencing predictions.
This helps advisors understand **why** the model predicted a certain risk, not just the probability.
