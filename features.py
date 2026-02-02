from datetime import date, datetime
import pandas as pd

FEATURE_COLUMNS = [
    "Age", "BusinessTravel", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "Gender", "JobInvolvement", "JobLevel",
    "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked", "OverTime",
    "PercentSalaryHike", "PerformanceRating",
    "TotalWorkingYears", "WorkLifeBalance",
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
        "BusinessTravel": daily["BusinessTravel"],
        "DistanceFromHome": 10,
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
        "TotalWorkingYears": emp["TotalWorkingYears"],
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
