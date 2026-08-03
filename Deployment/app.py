import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page Configuuration
st.set_page_config(page_title="Staff Attrition Predictor", page_icon="👥", layout="centered")

# Load Model Artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    with open(os.path.join(BASE_DIR, "attrition_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, "model_columns.pkl"), "rb") as f:
        model_columns = pickle.load(f)
    return model, scaler, model_columns

model, scaler, model_columns = load_artifacts()

st.title("👥 Staff Attrition Predictor")
st.markdown(
    "Estimate the likelihood that an employee will leave the company, "
    "based on job, compensation, and satisfaction factors."
)
st.divider()

# Input Form 
with st.form("attrition_form"):

    with st.expander("🧍 Personal Information", expanded=True):
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education = st.selectbox(
            "Education Level",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1: "Below College", 2: "College", 3: "Bachelor",
                                     4: "Master", 5: "Doctor"}[x]
        )
        education_field = st.selectbox(
            "Education Field",
            ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
        )
        distance_from_home = st.slider("Distance From Home (km)", 1, 30, 10)

    with st.expander("💼 Job Information"):
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox(
            "Job Role",
            ["Sales Executive", "Research Scientist", "Laboratory Technician",
             "Manufacturing Director", "Healthcare Representative", "Manager",
             "Sales Representative", "Research Director", "Human Resources"]
        )
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        business_travel = st.selectbox(
            "Business Travel Frequency", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
        )
        num_companies_worked = st.slider("Number of Companies Worked At", 0, 10, 2)
        total_working_years = st.slider("Total Working Years", 0, 40, 8)

    with st.expander("💰 Compensation"):
        monthly_income = st.number_input("Monthly Income (₦)", min_value=1000, max_value=50000, value=5000, step=100)
        daily_rate = st.slider("Daily Rate", 100, 1500, 800)
        hourly_rate = st.slider("Hourly Rate", 30, 100, 65)
        monthly_rate = st.slider("Monthly Rate", 2000, 27000, 14000)
        percent_salary_hike = st.slider("Percent Salary Hike (last review)", 10, 25, 15)
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])

    with st.expander("😊 Work Conditions & Satisfaction"):
        overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
        environment_satisfaction = st.select_slider("Environment Satisfaction", options=[1, 2, 3, 4], value=3)
        job_satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=3)
        job_involvement = st.select_slider("Job Involvement", options=[1, 2, 3, 4], value=3)
        relationship_satisfaction = st.select_slider("Relationship Satisfaction", options=[1, 2, 3, 4], value=3)
        work_life_balance = st.select_slider("Work-Life Balance", options=[1, 2, 3, 4], value=3)
        performance_rating = st.selectbox("Performance Rating", [3, 4])
        training_times_last_year = st.slider("Training Sessions Last Year", 0, 6, 2)

    with st.expander("📅 Tenure Details"):
        years_at_company = st.slider("Years At Company", 0, 40, 5)
        years_in_current_role = st.slider("Years In Current Role", 0, 20, 3)
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        years_with_curr_manager = st.slider("Years With Current Manager", 0, 20, 3)

    submitted = st.form_submit_button("🔍 Predict Attrition Risk", use_container_width=True)

# Prediction Logic 
if submitted:
    # Build raw input dict matching original dataset column names
    input_dict = {
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    }

    input_df = pd.DataFrame([input_dict])

    # Replicate training-time preprocessing exactly 

    # 1. Ordinal encode BusinessTravel (same mapping as training)
    travel_order = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    input_df["BusinessTravel"] = input_df["BusinessTravel"].map(travel_order)

    # 2. One-hot encode nominal categoricals (same columns as training)
    nominal_cols = ["Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
    input_df = pd.get_dummies(input_df, columns=nominal_cols)

    # 3. Build the OverTime x Income interaction term
    # get_dummies without drop_first here, so reindex+fillna(0) below handles alignment safely
    overtime_col = "OverTime_Yes" if "OverTime_Yes" in input_df.columns else None
    if overtime_col:
        input_df["OverTime_Income_Interaction"] = input_df[overtime_col] * input_df["MonthlyIncome"]
    else:
        input_df["OverTime_Income_Interaction"] = 0

    # 4. Align columns exactly to what the model expects (fills any missing one-hot cols with 0)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # 5. Scale using the SAME scaler fitted during training
    input_scaled = scaler.transform(input_df)

    # 6. Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Attrition Risk — {probability*100:.1f}% likelihood of leaving")
    else:
        st.success(f"✅ Low Attrition Risk — {probability*100:.1f}% likelihood of leaving")

    st.progress(min(int(probability * 100), 100))

    with st.expander("ℹ️ What influences this prediction?"):
        st.markdown(
            "This model was built on the IBM HR Analytics Attrition dataset. "
            "Key factors found to influence attrition include:\n"
            "- **Overtime combined with income level** — overtime is far riskier for lower earners\n"
            "- **Income relative to job level** — being paid below your level's average raises risk\n"
            "- **Distance from home** — adds a modest, fairly constant risk regardless of income\n"
            "- **Department and marital status** — Sales and single employees show higher attrition independent of other factors"
        )