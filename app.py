import os
import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Load Dataset directly from the official web URL so you don't even need to upload diabetes.csv
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# 2. Preprocess Data
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scale and Train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Predictor function
def predict_diabetes(pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age):
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        return f"⚠️ High Risk: The model predicts a DIABETES risk.\nConfidence: {probability*100:.1f}%"
    else:
        return f"✅ Low Risk: The model predicts NO DIABETES.\nConfidence: {(1-probability)*100:.1f}%"

# 5. Gradio Interface Layout
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies", value=0),
        gr.Slider(minimum=0, maximum=200, value=120, label="Glucose Level (mg/dL)"),
        gr.Slider(minimum=0, maximum=150, value=70, label="Blood Pressure (mm Hg)"),
        gr.Slider(minimum=0, maximum=100, value=20, label="Skin Thickness (mm)"),
        gr.Slider(minimum=0, maximum=900, value=80, label="Insulin Level (mu U/ml)"),
        gr.Slider(minimum=0.0, maximum=60.0, value=25.0, label="BMI"),
        gr.Slider(minimum=0.0, maximum=2.5, value=0.5, label="Diabetes Pedigree Function"),
        gr.Slider(minimum=1, maximum=120, value=33, label="Age (years)")
    ],
    outputs=gr.Textbox(label="Assessment Result"),
    title="🩸 AI Diabetes Risk Assessment Agent",
    description="Input patient health metrics below to evaluate diabetes risk using Machine Learning.",
    theme="soft"
)

# 6. Production Port Binding
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    interface.launch(server_name="0.0.0.0", server_port=port)
