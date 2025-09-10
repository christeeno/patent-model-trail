# streamlit_app.py
import streamlit as st
import pandas as pd
import json
from typing import Dict, Any
import ml

st.set_page_config(page_title="Student Health Input", layout="centered")

st.title("Student Health Input Form")
st.markdown(
    "Enter the student's data below. This app displays your input and a simple heuristic health-risk suggestion (for demo purposes only)."
)

def predict_health_risk(row: Dict[str, Any]) -> str:
    """
    Simple heuristic to suggest a Health Risk Level.
    NOTE: This is NOT medical advice. It's a demonstration heuristic only.
    Heuristic rules (demo):
      - If any critical sign is beyond threshold -> 'High'
      - Else if some borderline values -> 'Moderate'
      - Else -> 'Low'
    Thresholds chosen for example only.
    """
    # Delegate prediction to ml.py model
    return ml.predict_health_risk(row)


with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        student_id = st.text_input("Student_ID", value="")
        age = st.number_input("Age", min_value=0, max_value=150, value=20, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other", "Prefer not to say"])
        heart_rate = st.number_input("Heart_Rate (bpm)", min_value=0.0, value=70.0, step=0.1, format="%.1f")
        blood_pressure_systolic = st.number_input(
            "Blood_Pressure_Systolic (mmHg)", min_value=0.0, value=120.0, step=0.1, format="%.1f"
        )

    with col2:
        blood_pressure_diastolic = st.number_input(
            "Blood_Pressure_Diastolic (mmHg)", min_value=0.0, value=80.0, step=0.1, format="%.1f"
        )
        stress_level_biosensor = st.number_input(
            "Stress_Level_Biosensor (0-10)", min_value=0.0, max_value=10.0, value=3.0, step=0.1, format="%.1f"
        )
        stress_level_self_report = st.number_input(
            "Stress_Level_Self_Report (0-10)", min_value=0.0, max_value=10.0, value=2.0, step=0.1, format="%.1f"
        )
        physical_activity = st.selectbox(
            "Physical_Activity", options=["High", "Moderate", "Low", "None"]
        )

    sleep_quality = st.selectbox("Sleep_Quality", options=["Good", "Fair", "Poor", "Very Poor"])
    mood = st.selectbox("Mood", options=["Happy", "Neutral", "Anxious", "Sad", "Irritable", "Other"])
    study_hours = st.number_input("Study_Hours (per day)", min_value=0.0, value=3.0, step=0.1, format="%.1f")
    project_hours = st.number_input("Project_Hours (per day)", min_value=0.0, value=1.0, step=0.1, format="%.1f")

    # Allow user to optionally provide Health_Risk_Level (maybe from external label). If left blank, we will predict.
    provided_health_risk = st.selectbox(
        "Health_Risk_Level (optional - if you have a label choose it; else select 'Auto')",
        options=["Auto", "Low", "Moderate", "High"],
        index=0,
    )

    submitted = st.form_submit_button("Submit")

if submitted:
    # Build a dict / row to display
    # Encode categorical values to match ML model expectations
    gender_map = {"Male": 1, "Female": 0, "Other": 2, "Prefer not to say": 3}
    activity_map = {"High": 0, "Moderate": 1, "Low": 2, "None": 3}
    sleep_map = {"Good": 0, "Fair": 1, "Poor": 2, "Very Poor": 3}
    mood_map = {"Happy": 0, "Neutral": 1, "Anxious": 2, "Sad": 3, "Irritable": 4, "Other": 5}
    row = {
        "Student_ID": student_id,
        "Age": int(age),
        "Gender": gender_map.get(gender, 3),
        "Heart_Rate": float(heart_rate),
        "Blood_Pressure_Systolic": float(blood_pressure_systolic),
        "Blood_Pressure_Diastolic": float(blood_pressure_diastolic),
        "Stress_Level_Biosensor": float(stress_level_biosensor),
        "Stress_Level_Self_Report": float(stress_level_self_report),
        "Physical_Activity": activity_map.get(physical_activity, 3),
        "Sleep_Quality": sleep_map.get(sleep_quality, 3),
        "Mood": mood_map.get(mood, 5),
        "Study_Hours": float(study_hours),
        "Project_Hours": float(project_hours),
    }

    # Use ML model for prediction if Auto, else use user-provided label
    if provided_health_risk != "Auto":
        final_health_risk = provided_health_risk
        note = "User-provided label"
    else:
        # Call ml.py model prediction
        final_health_risk = ml.predict_health_risk(row)
        note = "ML model prediction"

    row["Health_Risk_Level"] = final_health_risk

    st.success("Input received ✔️")
    st.subheader("Entered values (table)")
    df = pd.DataFrame([row])
    st.dataframe(df)

    st.subheader("Summary (text output)")
    summary_lines = [
        f"Student ID: {row['Student_ID'] or '(not provided)'}",
        f"Age: {row['Age']}",
        f"Gender: {row['Gender']}",
        f"Heart Rate: {row['Heart_Rate']} bpm",
        f"Blood Pressure: {row['Blood_Pressure_Systolic']}/{row['Blood_Pressure_Diastolic']} mmHg (Systolic/Diastolic)",
        f"Stress (biosensor/self-report): {row['Stress_Level_Biosensor']}/{row['Stress_Level_Self_Report']}",
        f"Physical Activity: {row['Physical_Activity']}",
        f"Sleep Quality: {row['Sleep_Quality']}",
        f"Mood: {row['Mood']}",
        f"Study Hours: {row['Study_Hours']} per day",
        f"Project Hours: {row['Project_Hours']} per day",
        f"Health Risk Level: {row['Health_Risk_Level']}  —  ({note})",
    ]
    st.write("\n".join(summary_lines))

    st.subheader("JSON output")
    st.code(json.dumps(row, indent=2))

    st.markdown(f"<h2 style='color:blue;'>Predicted Health Risk Level: <span style='color:red;'>{row['Health_Risk_Level']}</span></h2>", unsafe_allow_html=True)
else:
    st.info("Fill the form and click Submit to display the text output and summary.")
