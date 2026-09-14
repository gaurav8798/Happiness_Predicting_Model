import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Happiness Predictor",
    page_icon="😊",
    layout="centered"
)

model = joblib.load("happiness_people_model.pkl")

df = pd.read_csv("Happy_peoples_.csv")

st.set_page_config(
    page_title="Happiness Predictor",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Happiness Predictor")
st.caption("Machine Learning based lifestyle prediction")

st.info(
    "This app uses a trained Random Forest model to predict "
    "whether a person is actually happy based on lifestyle factors."
)

st.write(
    "Enter your lifestyle details below and let the ML model "
    "predict whether the person is actually happy."
)

st.divider()

st.subheader("👤 Lifestyle Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=15,
        max_value=80,
        value=25
    )

    job = st.selectbox(
        "Job",
        sorted(df["Job"].dropna().unique())
    )

    work_hours = st.number_input(
        "Work Hours",
        min_value=0.0,
        max_value=24.0,
        value=8.0
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0
    )

    wake_up_time = st.selectbox(
        "Wake Up Time",
        sorted(df["Wake_Up_Time"].dropna().unique())
    )

    exercise_hours = st.number_input(
        "Exercise Hours",
        min_value=0.0,
        max_value=12.0,
        value=1.0
    )

with col2:
    family_time = st.number_input(
        "Family Time Hours",
        min_value=0.0,
        max_value=12.0,
        value=2.0
    )

    social_time = st.number_input(
        "Social Time Hours",
        min_value=0.0,
        max_value=12.0,
        value=2.0
    )

    meditation_hours = st.number_input(
        "Meditation Hours",
        min_value=0.0,
        max_value=12.0,
        value=0.5
    )

    stress_level = st.number_input(
        "Stress Level",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )

    life_satisfaction = st.number_input(
        "Life Satisfaction",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )

if st.button("Predict Happiness", type="primary"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Job": [job],
        "Work_Hours": [work_hours],
        "Sleep_Hours": [sleep_hours],
        "Wake_Up_Time": [wake_up_time],
        "Exercise_Hours": [exercise_hours],
        "Family_Time_Hours": [family_time],
        "Social_Time_Hours": [social_time],
        "Meditation_Hours": [meditation_hours],
        "Stress_Level": [stress_level],
        "Life_Satisfaction": [life_satisfaction]
    })

    prediction = model.predict(input_data)[0]

    # Get probabilities
    probabilities = model.predict_proba(input_data)[0]

    yes_index = list(model.classes_).index("Yes")
    yes_probability = probabilities[yes_index]

    st.divider()

    st.subheader("🔮 Prediction Result")

    if prediction == "Yes":
        st.success("😊 Actually Happy: YES")
    else:
        st.error("😔 Actually Happy: NO")

    st.metric(
        "Probability of being happy",
        f"{yes_probability:.1%}"
    )

    st.progress(float(yes_probability))