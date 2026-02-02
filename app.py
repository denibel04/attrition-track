import streamlit as st
import json
import joblib
from datetime import date, datetime
import pandas as pd
import xgboost 

# Load data
with open("data/employees.json", "r", encoding="utf-8") as f:
    employees = json.load(f)

# Load model
model = joblib.load("model/employee_attrition.joblib")

# Initialize session state
st.session_state.setdefault("page", "login")
st.session_state.setdefault("selected_employee", None)

# -------------------------------
# Callback functions
# -------------------------------
def select_employee(emp):
    st.session_state.selected_employee = emp
    st.session_state.page = "home"

def logout():
    st.session_state.selected_employee = None
    st.session_state.page = "login"

# -------------------------------
# Login page
# -------------------------------
if st.session_state.page == "login":
    st.title("Sign in as...")
    cols = st.columns(3)
    for i, emp in enumerate(employees):
        col = cols[i % 3]
        with col:
            with st.container(border=True, horizontal_alignment="center"):
                avatar_url = (
                    f"https://ui-avatars.com/api/?name="
                    f"{emp['FirstName']}+{emp['LastName']}"
                    f"&background=random&size=128&rounded=true"
                )

                st.image(avatar_url, width=100)
                st.markdown(
                    f"<p style='text-align:center;font-size:25px;font-weight:bold'>"
                    f"{emp['FirstName']} {emp['LastName']}</p>",
                    unsafe_allow_html=True,
                )
                st.write(emp["JobRole"])
                st.write(emp["Department"])

                st.button(
                    "Select",
                    key=f"select_{emp['id']}",
                    on_click=select_employee,
                    args=(emp,),
                )

# -------------------------------
# Home page
# -------------------------------
elif st.session_state.page == "home":
    emp = st.session_state.selected_employee
    avatar_url = (
        f"https://ui-avatars.com/api/?name="
        f"{emp['FirstName']}+{emp['LastName']}"
        f"&background=random&size=128&rounded=true"
    )
    st.image(avatar_url, width=100)
    st.title(f"Welcome, {emp['FirstName']} {emp['LastName']}!")

    ENV_SAT = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
    JOB_INV = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
    JOB_SAT = {1: "Very Low", 2: "Low", 3: "High", 4: "Very High"}
    PERFORMANCE = {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"}
    WLB = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}
    OVERTIME = {0: "No", 1: "Yes"}

    with st.form("daily_form"):
        st.subheader("Workday evaluation")

        col1, col2 = st.columns(2)

        with col1:
            env_sat = st.selectbox(
                "Environment Satisfaction",
                options=list(ENV_SAT.keys()),
                format_func=lambda x: ENV_SAT[x],
            )

            job_inv = st.selectbox(
                "Job Involvement",
                options=list(JOB_INV.keys()),
                format_func=lambda x: JOB_INV[x],
            )

            job_sat = st.selectbox(
                "Job Satisfaction",
                options=list(JOB_SAT.keys()),
                format_func=lambda x: JOB_SAT[x],
            )

        with col2:
            overtime = st.selectbox(
                "Overtime",
                options=list(OVERTIME.keys()),
                format_func=lambda x: OVERTIME[x],
            )

            performance = st.selectbox(
                "Performance Rating",
                options=list(PERFORMANCE.keys()),
                format_func=lambda x: PERFORMANCE[x],
            )

            wlb = st.selectbox(
                "Work Life Balance",
                options=list(WLB.keys()),
                format_func=lambda x: WLB[x],
            )

        submitted = st.form_submit_button("Submit")

        if submitted:
            record = {
                "date": str(date.today()),
                "employee_id": emp["id"],
                "EnvironmentSatisfaction": env_sat,
                "JobInvolvement": job_inv,
                "JobSatisfaction": job_sat,
                "OverTime": overtime,
                "PerformanceRating": performance,
                "WorkLifeBalance": wlb,
            }

            st.success("Form submitted!")
            st.write(record)


    FEATURE_COLUMNS = [
        "Age", "BusinessTravel", "DistanceFromHome", "Education",
        "EnvironmentSatisfaction", "Gender", "JobInvolvement", "JobLevel",
        "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked", "OverTime",
        "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
        "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "Department_Research & Development", "Department_Sales",
        "EducationField_Life Sciences", "EducationField_Marketing",
        "EducationField_Medical", "EducationField_Other",
        "EducationField_Technical Degree",
        "JobRole_Human Resources", "JobRole_Laboratory Technician",
        "JobRole_Manager", "JobRole_Manufacturing Director",
        "JobRole_Research Director", "JobRole_Research Scientist",
        "JobRole_Sales Executive", "JobRole_Sales Representative",
        "MaritalStatus_Married", "MaritalStatus_Single"
    ]

    

    def years_between(d):
        return (date.today() - d).days // 365

    def parse(d): 
        return datetime.strptime(d, "%Y-%m-%d").date()

    def build_input(emp, daily):
        age = years_between(parse(emp["BirthDate"]))
        years_at_company = years_between(parse(emp["ContractStartDate"]))
        years_in_role = years_between(parse(emp["CurrentRoleStartDate"]))
        years_since_promo = years_between(parse(emp["LastPromotionDate"]))

        data = {
            "Age": age,
            "BusinessTravel": emp["BusinessTravel"],
            "DistanceFromHome": 10,  # placeholder
            "Education": emp["Education"],
            "EnvironmentSatisfaction": daily["EnvironmentSatisfaction"],
            "Gender": emp["Gender"],
            "JobInvolvement": daily["JobInvolvement"],
            "JobLevel": emp["JobLevel"],
            "JobSatisfaction": daily["JobSatisfaction"],
            "MonthlyIncome": emp["MonthlyIncome"],
            "NumCompaniesWorked": emp["NumCompaniesWorked"],
            "OverTime": daily["OverTime"],
            "PercentSalaryHike": emp["PercentSalaryHike"],
            "PerformanceRating": daily["PerformanceRating"],
            "RelationshipSatisfaction": emp["RelationshipSatisfaction"],
            "TotalWorkingYears": emp["TotalWorkingYears"],
            "TrainingTimesLastYear": emp["TrainingTimesLastYear"],
            "WorkLifeBalance": daily["WorkLifeBalance"],
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promo,
            "YearsWithCurrManager": emp["YearsWithCurrManager"],

            # ---- DUMMIES (drop_first applied) ----
            "Department_Research & Development": 1 if emp["Department"] == "Research & Development" else 0,
            "Department_Sales": 1 if emp["Department"] == "Sales" else 0,

            "EducationField_Life Sciences": 1 if emp["EducationField"] == "Life Sciences" else 0,
            "EducationField_Marketing": 1 if emp["EducationField"] == "Marketing" else 0,
            "EducationField_Medical": 1 if emp["EducationField"] == "Medical" else 0,
            "EducationField_Other": 1 if emp["EducationField"] == "Other" else 0,
            "EducationField_Technical Degree": 1 if emp["EducationField"] == "Technical Degree" else 0,

            "JobRole_Human Resources": 1 if emp["JobRole"] == "Human Resources" else 0,
            "JobRole_Laboratory Technician": 1 if emp["JobRole"] == "Laboratory Technician" else 0,
            "JobRole_Manager": 1 if emp["JobRole"] == "Manager" else 0,
            "JobRole_Manufacturing Director": 1 if emp["JobRole"] == "Manufacturing Director" else 0,
            "JobRole_Research Director": 1 if emp["JobRole"] == "Research Director" else 0,
            "JobRole_Research Scientist": 1 if emp["JobRole"] == "Research Scientist" else 0,
            "JobRole_Sales Executive": 1 if emp["JobRole"] == "Sales Executive" else 0,
            "JobRole_Sales Representative": 1 if emp["JobRole"] == "Sales Representative" else 0,

            "MaritalStatus_Married": 1 if emp["MaritalStatus"] == "Married" else 0,
            "MaritalStatus_Single": 1 if emp["MaritalStatus"] == "Single" else 0,
        }

        return pd.DataFrame([data])[FEATURE_COLUMNS]

    X = build_input(emp, record)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[0][1]

    st.subheader("Attrition Prediction")
    st.write(f"y_pred: {y_pred[0]}")
    st.write(f"y_prob: {y_prob}")

    st.button("Log out", on_click=logout)
