import streamlit as st
import json
import joblib
from datetime import date, datetime
import pandas as pd
import xgboost 
from features import build_input, years_between
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


st.set_page_config(
    page_title="Work Satisfaction Tracker",
    page_icon="💼",
)

# Load data
if "employees_session" not in st.session_state:
    with open("data/employees.json", "r", encoding="utf-8") as f:
        st.session_state.employees_session = json.load(f)

with open("data/history.json", "r", encoding="utf-8") as f:
    all_scores = json.load(f)

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

def create_employee():
    st.session_state.page = "create_employee"

def logout():
    st.session_state.selected_employee = None
    st.session_state.page = "login"

# -------------------------------
# Login page
# -------------------------------
if st.session_state.page == "login":

    st.title("😸 Work Satisfaction Tracker")
    st.image("img/banner.jpg")

    st.subheader("Sign in as...")
    st.button("Create new employee", on_click=create_employee, icon=":material/person_add:", type="primary")

    cols = st.columns(3)
    for i, emp in enumerate(st.session_state.employees_session):
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
                    type="primary",
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
    BUSINESS_TRAVEL_OPTIONS = {0: "No Travel", 1: "Rarely", 2: "Frequently"}

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
            business_travel = st.selectbox(
                "Business Travel",
                options=list(BUSINESS_TRAVEL_OPTIONS.keys()),
                format_func=lambda x: BUSINESS_TRAVEL_OPTIONS[x]
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

        submitted = st.form_submit_button("Submit", type="primary", icon=":material/send:")

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
                "BusinessTravel": business_travel
            }

            st.success("Form submitted!")

            X = build_input(emp, record)
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[0][1]

            # st.subheader("Attrition Prediction")
            # st.write(f"y_pred: {y_pred[0]}")
            # st.write(f"y_prob: {y_prob}")

            emp_scores = [s for s in all_scores if s["id"] == emp["id"]]
            emp_scores.append({
                "id": emp["id"],
                "date": str(date.today()),
                "score": y_prob
            })
            emp_scores = sorted(emp_scores, key=lambda x: x["date"])

            dates = [datetime.strptime(s["date"], "%Y-%m-%d") for s in emp_scores]
            scores = [1 - s["score"] for s in emp_scores]  
            scores_pct = [v*100 for v in scores]

            if len(emp_scores) == 1:
                st.metric(
                label="Work Satisfaction",
                value=f"{scores_pct[0]:.1f}%",
                delta=None 
                )

            else:
                fig, ax = plt.subplots(figsize=(6,4))
                
                ax.plot(dates, scores_pct, marker='o', linestyle='--', color='#8070adff') 
                for i, v in enumerate(scores_pct):
                    ax.text(dates[i], v + 4, f"{v:.1f}%", ha='center') 

                # Format x-axis to show only one label per month
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))

                # Remove y-axis ticks
                ax.set_yticks([])

                ax.set_title("Work Satisfaction Over Time")
                ax.set_ylabel("Work Satisfaction (%)")
                ax.set_ylim(0, 100)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)



    st.button("Log out", on_click=logout, type="primary", icon=":material/logout:")

# -------------------------------
# Create employee page
# -------------------------------
elif st.session_state.page == "create_employee":

    st.button("Go back", on_click=logout, type="primary", icon=":material/arrow_back_ios:")

    st.title("Create new employee")

    EDUCATION_OPTIONS = {
    1: "Below College",
    2: "College",
    3: "Bachelor",
    4: "Master",
    5: "Doctor"
    }

    JOB_LEVELS = {1: "Entry", 2: "Low", 3: "Medium", 4: "High", 5: "Top"}

    today = date.today()
    earliest_birth = date(1950, 1, 1)
    latest_birth = date(today.year - 18, today.month, today.day)
    latest_contract_start = today

    with st.form("create_employee_form"):
        col1, col2 = st.columns(2)

        with col1:
            first_name = st.text_input("First name")
            last_name = st.text_input("Last name")
            birth_date = st.date_input("Birth date", min_value=earliest_birth, max_value=latest_birth, value=date(1990, 1, 1))
            gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
            home_address = st.text_input("Home address")

            education = st.selectbox(
                "Education",
                options=list(EDUCATION_OPTIONS.keys()),
                format_func=lambda x: EDUCATION_OPTIONS[x]
            )
            job_level = st.selectbox(
                "Job level",
                options=list(JOB_LEVELS.keys()),
                format_func=lambda x: JOB_LEVELS[x]
            )

        with col2:
           
            num_companies = st.number_input("Number of companies worked", min_value=0)

            contract_start = st.date_input("Contract start date", min_value=date(1970, 1, 1), max_value=today)
            role_start = st.date_input("Current role start date", min_value=date(1970, 1, 1), max_value=today)
            last_promo = st.date_input("Last promotion date", min_value=date(1970, 1, 1), max_value=today)
            last_manager_date = st.date_input("Last manager date", min_value=date(1970, 1, 1), max_value=today)

            department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
            education_field = st.selectbox(
                "Education field",
                ["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"]
            )
            job_role = st.selectbox(
                "Job role",
                [
                    "Research Scientist",
                    "Laboratory Technician",
                    "Sales Executive",
                    "Sales Representative",
                    "Manager",
                    "Human Resources",
                ],
            )
        
        monthly_income = st.slider(
            "Monthly Salary",
            min_value=1000,
            max_value=20000,
            value=7500,
            step=100
        )

        percent_hike = st.slider(
            "Percent Salary Hike",
            min_value=0,
            max_value=100,
            value=15,
            step=1
        )

        submitted = st.form_submit_button("Create employee", type="primary")

    if submitted:
        if not first_name.strip() or not last_name.strip() or not home_address.strip():
            st.error("Please fill in First name, Last name, and Home address.")
        else:
            new_id = max(emp["id"] for emp in st.session_state.employees_session) + 1

            new_employee = {
                "id": new_id,
                "FirstName": first_name,
                "LastName": last_name,
                "BirthDate": birth_date.isoformat(),
                "Gender": gender,
                "MaritalStatus": marital_status,
                "HomeAddress": home_address,
                "Education": education,
                "JobLevel": job_level,
                "MonthlyIncome": monthly_income,
                "NumCompaniesWorked": num_companies,
                "PercentSalaryHike": percent_hike,
                "TotalWorkingYears": years_between(contract_start),
                "ContractStartDate": contract_start.isoformat(),
                "CurrentRoleStartDate": role_start.isoformat(),
                "LastPromotionDate": last_promo.isoformat(),
                "YearsWithCurrManager": years_between(last_manager_date),
                "Department": department,
                "EducationField": education_field,
                "JobRole": job_role,
            }
            st.session_state.employees_session.append(new_employee)
            st.success("Employee created successfully 🎉")
            # st.session_state.page = "login"
            # st.rerun()

st.markdown(
    """
    <hr>
    <div style="text-align:center; color:gray;">
        Developed by Jesús Herrera, Denisa Belean and Antonio Delgado © 2026
    </div>
    """,
    unsafe_allow_html=True
)


    






