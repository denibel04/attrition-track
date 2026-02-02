# Employee Attrition Prediction (Streamlit App)

This interactive **Streamlit** web application predicts the **probability of employee attrition** using a **Machine Learning model (XGBoost)**.  
The goal is to simulate monthly work evaluations and visualize how an employee’s risk of leaving the company evolves over time.

---

## Features

### Employee creation and selection
- Visual selection of existing employees using cards
- Creation of new employees directly from the interface

---

### Monthly Employee Evaluation
For each employee, a monthly evaluation can be submitted including:

- Environment satisfaction
- Job involvement
- Job satisfaction
- Performance rating
- Work-life balance
- Overtime
- Business travel

These daily inputs are combined with the employee’s static information to generate predictions.

---

### Attrition Prediction
- Model: **XGBoost Classifier**
- Outputs:
  - Binary prediction: *Stays / Leaves*
  - Attrition probability (%)
  - With attrition we calculate work satisfaction (100 - attrition probability)
- Visualization:
  - **Line chart** if historical predictions exist
  - **Bar chart** if only today’s prediction is available

---

### Results Visualization
- Charts built with **Matplotlib**
- Clear percentage labels
- Automatically adapts based on available history

---

### Employee Management
Create employees with:

- Personal information
- Education level and job level
- Salary and work experience
- Department, role, and education field
- Automatic calculation of experience years from dates

Employees are stored in browser temporal memory using **`st.session_state`** to simulate new employees.

Feature engineering is handled in `features.py`.

