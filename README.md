# 🩸 Diabetes Risk Assessment Agent

An end-to-end Machine Learning and AI workflow that preprocesses clinical health data, trains and compares multiple classification models, develops an Artificial Neural Network (ANN) extension, and deploys the highest-performing pipeline as a live, interactive web application using **Gradio**.

---

## 🚀 Features
*   **Robust Data Imputation:** Automatically detects and replaces invalid `0` entries (in columns like Glucose, BMI, and Insulin) with statistically sound medians.
*   **Feature Scaling Pipeline:** Integrates `StandardScaler` to ensure normalized feature inputs for reliable classification.
*   **Model Benchmarking Suite:** Trains, tests, and evaluates six prominent classification algorithms:
    *   Logistic Regression
    *   Random Forest Classifier
    *   XGBoost Classifier
    *   Support Vector Machine (SVM)
    *   K-Nearest Neighbors (KNN)
    *   Multi-Layer Perceptron (MLP) / Deep Learning ANN Extension
*   **Clinical-First Evaluation:** Evaluates systems utilizing Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
*   **Instant Cloud Deployment:** Deploys the final model directly out of Google Colab into a shareable web GUI.

---

## 📊 Dataset Structure
The project utilizes the **Pima Indians Diabetes Dataset** available on [Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set). The model maps eight diagnostic features to predict binary risk:

| Feature Metric | Description |
| :--- | :--- |
| **Pregnancies** | Number of times pregnant |
| **Glucose** | Plasma glucose concentration (2 hours in an oral glucose tolerance test) |
| **BloodPressure** | Diastolic blood pressure (mm Hg) |
| **SkinThickness** | Triceps skin fold thickness (mm) |
| **Insulin** | 2-Hour serum insulin (mu U/ml) |
| **BMI** | Body mass index (weight in kg / (height in m)²) |
| **DiabetesPedigreeFunction** | A score indicative of hereditary diabetes risk based on family history |
| **Age** | Age of the patient (years) |
| **Outcome** | Target class label (0 = Low Risk / Non-Diabetic, 1 = High Risk / Diabetic) |

---

## 🛠️ Step-by-Step Installation & Setup

Follow these simple steps to run the repository on your local machine or Google Colab.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/diabetes-risk-assessment-agent.git](https://github.com/YOUR_USERNAME/diabetes-risk-assessment-agent.git)
cd diabetes-risk-assessment-agent.
 ```
```
pip install pandas numpy scikit-learn xgboost gradio matplotlib seaborn
