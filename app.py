import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# ================== Page Config ==================
st.set_page_config(
    page_title="Personality Prediction System",
    page_icon="🧠",
    layout="wide"
)

# ================== Custom CSS ==================
st.markdown("""
<style>
/* Hide Streamlit default UI */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Light gradient background */
.stApp {
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
}

/* Glassmorphism cards */
.card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    animation: fadeIn 0.6s ease-in-out;
}

/* Fade animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Metric numbers */
.metric {
    font-size: 32px;
    font-weight: bold;
    color: #1e88e5;
}

/* Prediction result */
.result {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    color: #2e7d32;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #42a5f5, #478ed1);
    color: white;
    border-radius: 30px;
    padding: 0.6em 1.8em;
    font-weight: bold;
    font-size: 16px;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #1e88e5, #1565c0);
}
</style>
""", unsafe_allow_html=True)

# ================== Load Dataset ==================
@st.cache_data
def load_data():
    return pd.read_csv("personality_dataset.csv")

df = load_data()

target = "Personality"
X = df.drop(target, axis=1)
y = df[target]

# ================== Encode ==================
encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].str.lower())
        encoders[col] = le

le_y = LabelEncoder()
y = le_y.fit_transform(y)

# ================== Train Model ==================
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(X, y)
y_pred = model.predict(X_test)

# ================== Metrics ==================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

st.subheader("📊 Model Performance")

c1, c2, c3, c4 = st.columns(4)
for col, name, value in zip(
    [c1, c2, c3, c4],
    ["Accuracy", "Precision", "Recall", "F1 Score"],
    [accuracy, precision, recall, f1]
):
    with col:
        st.markdown(f"""
        <div class="card">
            <p>{name}</p>
            <p class="metric">{value:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

# ================== Confusion Matrix ==================
st.markdown("<br>", unsafe_allow_html=True)
labels = le_y.inverse_transform(np.arange(len(le_y.classes_)))
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues")
st.pyplot(fig)

# ================== Prediction Form ==================
st.subheader("🎯 Predict Personality")

with st.form("prediction_form"):
    Time_spent_Alone = st.slider("Time spent alone (hours/day)", 0, 24, 5)
    Stage_fear = st.selectbox("Stage fear", ["yes", "no"])
    Social_event_attendance = st.number_input("Social events per month", 0, 50, 2)
    Going_outside = st.slider("Going outside (times/week)", 0, 7, 3)
    Drained_after_socializing = st.selectbox("Drained after socializing", ["yes", "no"])
    Friends_circle_size = st.number_input("Friends circle size", 0, 100, 5)
    Post_frequency = st.number_input("Posts per week", 0, 50, 1)
    submit = st.form_submit_button("🔮 Predict")

# ================== PDF Generator ==================
def generate_pdf(input_data, prediction):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Personality Prediction Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated on: {datetime.now()}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    table_data = [["Feature", "Value"]]
    for k, v in input_data.items():
        table_data.append([k, str(v)])

    table_data.append(["Predicted Personality", prediction])

    table = Table(table_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("FONT", (0,0), (-1,0), "Helvetica-Bold")
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ================== Prediction + PDF ==================
if submit:
    input_df = pd.DataFrame([[
        Time_spent_Alone, Stage_fear, Social_event_attendance,
        Going_outside, Drained_after_socializing,
        Friends_circle_size, Post_frequency
    ]], columns=X.columns)

    for col in input_df.columns:
        if col in encoders:
            val = input_df[col].iloc[0].lower()
            input_df[col] = encoders[col].transform([val])

    prediction = model.predict(input_df)
    result = le_y.inverse_transform(prediction)[0]

    st.markdown(f"""
    <div class="card result">
        🧠 Predicted Personality<br>{result}
    </div>
    """, unsafe_allow_html=True)

    pdf = generate_pdf(input_df.iloc[0].to_dict(), result)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf,
        file_name="personality_prediction_report.pdf",
        mime="application/pdf"
    )
