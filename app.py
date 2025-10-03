import streamlit as st # type: ignore
import pandas as pd
import joblib

# Load your pre-trained RandomForest model
model = joblib.load('./models/random_forest_model.joblib')

st.set_page_config(page_title="Student GradeClass Predictor", layout="centered")

st.title("🎓 Student Performance GradeClass Prediction")

st.image("./img/book.png", caption="Study Time Matters!", width=300)

with st.sidebar:
    st.header("Student Information")

    age = st.slider("Age", min_value=15, max_value=18, value=16, help="Select student's age")
    gender = st.radio(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "Male" if x == 0 else "Female",
        index=0,
        help="Select student's gender"
    )
    ethnicity = st.selectbox(
        "Ethnicity",
        options=[0, 1, 2, 3],
        index=2,
        format_func=lambda x: ["Caucasian", "African American", "Asian", "Other"][x],
        help="Select student's ethnicity"
    )
    parent_education = st.selectbox(
        "Parental Education Level",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: ["None", "High School", "Some College", "Bachelor's", "Higher"][x],
        help="Select education level of student's parents"
    )

    st.markdown("---")
    st.header("Study & Attendance")

    study_time = st.number_input(
        "Weekly Study Time (hours)",
        min_value=0,
        max_value=20,
        value=10,
        step=1,
        help="Enter the average hours spent studying weekly"
    )
    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=30,
        value=3,
        step=1,
        help="Enter the total number of absences"
    )
    tutoring = st.radio(
        "Receives Tutoring?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Does the student receive tutoring?"
    )

    st.markdown("---")
    st.header("Parental Involvement")

    parental_support = st.select_slider(
        "Parental Support Level",
        options=[0, 1, 2, 3, 4],
        value=2,
        format_func=lambda x: ["None", "Low", "Moderate", "High", "Very High"][x],
        help="Level of parental support for the student"
    )

    st.markdown("---")
    st.header("Extracurricular Activities")

    extracurricular = st.checkbox("Participates in Extracurricular Activities")
    sports = st.checkbox("Participates in Sports")
    music = st.checkbox("Participates in Music")
    volunteering = st.checkbox("Participates in Volunteering")

    st.markdown("---")
    st.header("Academic Performance")

    gpa = st.slider("GPA", 2.0, 4.0, 3.0, step=0.01, help="Grade Point Average (scale 2.0 to 4.0)")

# Convert boolean checkboxes to 0/1 for model input
extracurricular = int(extracurricular)
sports = int(sports)
music = int(music)
volunteering = int(volunteering)

# Prepare input dataframe matching model's expected input
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Ethnicity': [ethnicity],
    'ParentalEducation': [parent_education],
    'StudyTimeWeekly': [study_time],
    'Absences': [absences],
    'Tutoring': [tutoring],
    'ParentalSupport': [parental_support],
    'Extracurricular': [extracurricular],
    'Sports': [sports],
    'Music': [music],
    'Volunteering': [volunteering],
    'GPA': [gpa]
})

if st.button("Predict GradeClass"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df).max()
    grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
    st.balloons()
    st.success(f"✅ Predicted GradeClass: **{grade_map[prediction]}** (Confidence: {proba:.2f})")
   
