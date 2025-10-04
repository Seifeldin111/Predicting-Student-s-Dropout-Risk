import streamlit as st


st.title(" 🎓 Student Dropout Risk Prediction App")

st.markdown("""
    Welcome to the Student Dropout Risk Prediction App! This app helps advisors, teachers, 
    and administrators identify students who are at risk of dropping out, so that timely 
    interventions can be made.

    ### How the App Works
    This app consists of **two prediction modules**:

    1. **Prediction After 1st Semester**
       - Uses student features and the performance of the **first semester only**.
       - Helps advisors make early decisions about student support.
       - Input features include personal info, qualifications, and 1st semester grades.

    2. **Prediction After 2nd Semester**
       - Uses features and performance **after the 2nd semester**.
       - Provides a more complete risk assessment based on both semesters.

    ### How to Use
    1. Select the module you want to use from the sidebar.
    2. Fill in the student's information in the provided fields.
    3. Click **Submit** to get the predicted dropout probability.

    > ⚠️ Note: The predictions are based on historical data and statistical models. They 
    should support decision-making, not replace professional judgment.
    """)
