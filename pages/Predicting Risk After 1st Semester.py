import re
import streamlit as st
import pandas as pd
import joblib
import shap
import skops.io as sio
import skops
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
pip install imbalanced-learn



def load_options(file):
    df = pd.read_csv(file)
    return df.iloc[:, 1].dropna().unique().tolist()

options_yes_no = load_options("tA10_yes_no.csv")
options_marital = load_options("tA1_marital_status.csv")
options_application = load_options("tA3_application_mode.csv")
options_course = load_options("tA4_course_names.csv")
options_prev_quals = load_options("tA5_previous_quals.csv")
options_parent_quals = load_options("tA6_parent_previous_quals.csv")
options_parent_occ = load_options("tA7_parent_occupation.csv")
options_gender = load_options("tA8_gender.csv")
options_attendance = load_options("tA9_attendance_regime.csv")

#st.title("Student Dropout Risk Assessment After Semester 1")

st.markdown("<h1 style='color: #00008B;'>Student Dropout Risk Assessment After Semester 1</h1>", unsafe_allow_html=True)
st.markdown("---")



tab1, tab2, tab3, tab4 = st.tabs(["Student Info", "Family Background", "Academic Info", "Economic Info"])

with tab1:
    age = st.number_input("Age at enrollment", min_value=15, max_value=80, step=1)
    gender = st.selectbox("Gender", ["-- Select --"] + options_gender, index=0)
    previous_qualification = st.selectbox("Previous qualification", ["-- Select --"] + options_prev_quals, index=0)
    marital_status = st.selectbox("Marital status", ["-- Select --"] + options_marital, index=0)
    application_mode = st.selectbox("Application mode", ["-- Select --"] + options_application, index=0)
    application_order = st.number_input("Application order", min_value=1, step=1)
    edu_special_needs = st.selectbox("Educational special needs", ["-- Select --"] + options_yes_no, index=0)
    displaced = st.selectbox("Displaced", ["-- Select --"] + options_yes_no, index=0)
    debtor = st.selectbox("Debtor", ["-- Select --"] + options_yes_no, index=0)
    tuition_fees = st.selectbox("Tuition fees up to date", ["-- Select --"] + options_yes_no, index=0)
    scholarship = st.selectbox("Scholarship holder", ["-- Select --"] + options_yes_no, index=0)
    international = st.selectbox("International", ["-- Select --"] + options_yes_no, index=0)

with tab2:
    mother_qual = st.selectbox("Mother's qualification", ["-- Select --"] + options_parent_quals, index=0)
    father_qual = st.selectbox("Father's qualification", ["-- Select --"] + options_parent_quals, index=0)
    mother_occ = st.selectbox("Mother's occupation category", ["-- Select --"] + options_parent_occ, index=0)
    father_occ = st.selectbox("Father's occupation category", ["-- Select --"] + options_parent_occ, index=0)

with tab3:
      course_name = st.selectbox("Course", ["-- Select --"] + options_course, index=0)
      attendance = st.selectbox("Daytime/evening attendance", ["-- Select --"] + options_attendance, index=0)
      curr_units_1st_sem_credited = st.number_input("Curricular units 1st sem (credited)", min_value=0, step=1)
      curr_units_1st_sem_enrolled = st.number_input("Curricular units 1st sem (enrolled)", min_value=0, step=1)
      curr_units_1st_sem_evaluations = st.number_input("Curricular units 1st sem (evaluations)", min_value=0, step=1)
      curr_units_1st_sem_approved = st.number_input("Curricular units 1st sem (approved)", min_value=0, step=1)
      curr_units_1st_sem_grade = st.number_input("Curricular units 1st sem (grade)", min_value=0.0, step=0.1)
      curr_units_1st_sem_without_eval = st.number_input("Curricular units 1st sem (without evaluations)", min_value=0, step=1)

with tab4:
    unemployment_rate = st.number_input("Unemployment rate", step=0.1)
    inflation_rate = st.number_input("Inflation rate", step=0.1)
    gdp = st.number_input("GDP", step=0.1)






user_inputs = {
    "Marital status": marital_status,
    "Application mode": application_mode,
    "Application order": application_order,
    "Course": course_name,
    "Daytime/evening attendance": attendance,
    "Previous qualification": previous_qualification,
    "Mother's qualification": mother_qual,
    "Father's qualification": father_qual,
    "Mother's occupation": mother_occ,
    "Father's occupation": father_occ,
    "Displaced": displaced,
    "Educational special needs": edu_special_needs,
    "Debtor": debtor,
    "Tuition fees up to date": tuition_fees,
    "Gender": gender,
    "Scholarship holder": scholarship,
    "Age at enrollment": age,
    "International": international,
    "Curricular units 1st sem (credited)": curr_units_1st_sem_credited,
    "Curricular units 1st sem (enrolled)": curr_units_1st_sem_enrolled,
    "Curricular units 1st sem (evaluations)": curr_units_1st_sem_evaluations,
    "Curricular units 1st sem (approved)": curr_units_1st_sem_approved,
    "Curricular units 1st sem (grade)": curr_units_1st_sem_grade,
    "Curricular units 1st sem (without evaluations)": curr_units_1st_sem_without_eval,
    "Unemployment rate": unemployment_rate,
    "Inflation rate": inflation_rate,
    "GDP": gdp,
}

if all(v != "-- Select --" and v is not None for v in user_inputs.values()):
    if st.button("Submit"):
        pass_rate_sem1 = 0 if curr_units_1st_sem_enrolled == 0 else curr_units_1st_sem_approved / curr_units_1st_sem_enrolled
        economic_stress = unemployment_rate * (1 + inflation_rate)
        not_first_choice = 1 if application_order > 1 else 0
        def trim(x):
            if x in ["Single"]:
                return x
            elif x in ['Married', 'Common-law marriage']:
                return 'Married'
            else:
                return 'Separated'
        marital_status = trim(marital_status)

        if application_mode in ["1st phase—general contingent", "2nd phase—general contingent", "Over 23 years old", "Change in course", "Technological specialization diploma holders", "Holders of other higher courses", "3rd phase—general contingent", "Transfer", "Change in institution/course"]:
            pass
        else:
            application_mode = "Other"


        def qual_group(x):
            no_schooling = [
                "Cannot read or write",
                "Can read without having a 4th year of schooling",
                "Unknown"
            ]

            basic = [
                "Basic education 1st cycle (4th/5th year) or equivalent",
                "Basic education 2nd cycle (6th/7th/8th year) or equivalent",
                "Basic education 3rd cycle (9th/10th/11th year) or equivalent",
            ]

            secondary = [
                "Secondary education",
                "10th year of schooling",
                "10th year of schooling—not completed",
                "11th year of schooling—not completed",
                "12th year of schooling—not completed",
                "Other—11th year of schooling",
            ]

            technical = [
                "Professional higher technical course",
                "Technological specialization course"
            ]

            higher = [
                "Higher education—degree",
                "Higher education—degree (1st cycle)",
                "Higher education—bachelor’s degree",
                "Higher education—master’s degree",
                "Higher education—master’s degree (2nd cycle)",
                "Higher education—doctorate",
                "Frequency of higher education",
            ]

            if x in no_schooling:
                return "No Schooling / Illiterate"
            elif x in basic:
                return "Basic Education"
            elif x in secondary:
                return "Secondary Education"
            elif x in technical:
                return "Technical / Vocational"
            elif x in higher:
                return "Higher Education"
            else:
                return "Other"
        previous_qualification = qual_group(previous_qualification)


        def parents_qual_group(x):
            no_schooling = [
                "Cannot read or write",
                "Can read without having a 4th year of schooling",
                "Unknown"
            ]

            basic = [
                "Basic education 1st cycle (4th/5th year) or equivalent",
                "Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent",
                "Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent",
                "7th year of schooling",
                "8th year of schooling",
                "9th Year of Schooling—not completed",
                "7th Year (Old)"
            ]

            secondary = [
                "Secondary Education—12th Year of Schooling or Equivalent",
                "12th Year of Schooling—not completed",
                "2nd cycle of the general high school course",
                "10th Year of Schooling",
                "11th Year of Schooling—not completed",
                "Other—11th Year of Schooling",
                "2nd year complementary high school course",
                "Complementary High School Course",
                "Complementary High School Course—not concluded",
            ]

            technical = [
                "General Course of Administration and Commerce",
                "General commerce course",
                "Supplementary Accounting and Administration",
                "Technical-professional course",
                "Technological specialization course"
            ]

            higher = [
                "Higher Education—degree",
                "Higher Education—bachelor’s degree",
                "Higher Education—master’s degree",
                "Higher Education—doctorate",
                "Frequency of Higher Education"
            ]

            if x in no_schooling:
                return "No Schooling / Illiterate"
            elif x in basic:
                return "Basic Education"
            elif x in secondary:
                return "Secondary Education"
            elif x in technical:
                return "Technical / Vocational"
            elif x in higher:
                return "Higher Education"
            else:
                return "Other"
        mother_qual = parents_qual_group(mother_qual)
        father_qual = parents_qual_group(father_qual)


        def trim(x):
            unskilled = [
                "Unskilled Workers",
                "Meal preparation assistants",
                "Unskilled workers in extractive industry, construction, manufacturing, and transport",
                "Unskilled workers in agriculture, animal production, and fisheries and forestry",
                "Personal care workers and the like",
                "Street vendors (except food) and street service provider",
            ]

            skilled = [
                "Skilled Workers in Industry, Construction, and Craftsmen",
                "Installation and Machine Operators and Assembly Workers",
                "Vehicle drivers and mobile equipment operators",
                "Fixed plant and machine operators",
                "Assembly workers",
                "Skilled construction workers and the like, except electricians",
                "Skilled workers in metallurgy, metalworking, and similar",
                "Skilled workers in electricity and electronics",
                "Workers in food processing, woodworking, and clothing and other industries and crafts",
            ]

            services = [
                "Personal Services, Security and Safety Workers, and Sellers",
                "Personal service workers",
                "Sellers",
                "Protection and security services personnel",
                "Hotel, catering, trade, and other services directors",
            ]

            admin = [
                "Administrative staff",
                "Office workers, secretaries in general, and data processing operators",
                "Other administrative support staff",
                "Data, accounting, statistical, financial services, and registry-related operators",
            ]

            technicians = [
                "Intermediate Level Technicians and Professions",
                "Intermediate level science and engineering technicians and professions",
                "Technicians and professionals of intermediate level of health",
                "Intermediate level technicians from legal, social, sports, cultural, and similar services",
                "Information and communication technology technicians",
            ]

            professionals = [
                "Specialists in Intellectual and Scientific Activities",
                "Specialists in finance, accounting, administrative organization, and public and commercial relations",
                "Specialists in the physical sciences, mathematics, engineering, and related techniques",
                "Teachers",
                "Health professionals",
            ]

            managers = [
                "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
                "Directors of administrative and commercial services",
            ]

            armed_forces = [
                "Armed Forces Professions",
                "Armed Forces Officers",
                "Other Armed Forces personnel",
                "Armed Forces Sergeants",
            ]

            agriculture = [
                "Farmers and Skilled Workers in Agriculture, Fisheries, and Forestry",
                "Market-oriented farmers and skilled agricultural and animal production workers",
                "Farmers, livestock keepers, fishermen, hunters and gatherers, and subsistence",
            ]

            if x in unskilled:
                return "Unskilled"
            elif x in skilled:
                return "Skilled manual"
            elif x in services:
                return "Services"
            elif x in admin:
                return "Administrative"
            elif x in technicians:
                return "Technicians"
            elif x in professionals:
                return "Professionals"
            elif x in managers:
                return "Managers"
            elif x in armed_forces:
                return "Armed Forces"
            elif x in agriculture:
                return "Agriculture"
            else:
                return "Other"
        mother_occ = trim(mother_occ)
        father_occ = trim(father_occ)

        student_data = {
            "Application order": application_order,
            "Displaced": displaced,
            "Educational special needs": edu_special_needs,
            "Tuition fees up to date": tuition_fees,
            "Age at enrollment": age,
            "International": international,
            "Curricular units 1st sem (credited)": curr_units_1st_sem_credited,
            "Curricular units 1st sem (enrolled)": curr_units_1st_sem_enrolled,
            "Curricular units 1st sem (evaluations)": curr_units_1st_sem_evaluations,
            "Curricular units 1st sem (approved)": curr_units_1st_sem_approved,
            "Curricular units 1st sem (grade)": curr_units_1st_sem_grade,
            "Curricular units 1st sem (without evaluations)": curr_units_1st_sem_without_eval,
            "Unemployment rate": unemployment_rate,
            "Inflation rate": inflation_rate,
            "GDP": gdp,
            "pass_rate_sem1": pass_rate_sem1,
            "economic_stress": economic_stress,
            "not_first_choice": not_first_choice,
            "Marital status": marital_status,
            "Application mode": application_mode,
            "Course": course_name,
            "Daytime/evening attendance": attendance,
            "Previous qualification": previous_qualification,
            "Mother's qualification": mother_qual,
            "Father's qualification": father_qual,
            "Mother's occupation": mother_occ,
            "Father's occupation": father_occ,
            "Debtor": debtor,
            "Gender": gender,
            "Scholarship holder": scholarship,
        }

        student_df = pd.DataFrame([student_data])

        categorical_columns = student_df.select_dtypes(include=['object', 'category']).columns

        with open("pipeline.skops", "rb") as f:
            untrusted = skops.io.get_untrusted_types(data=None, file=f)

        model = sio.load("pipeline.skops", trusted=untrusted)

        with st.spinner("Calculating dropout probability..."):
            proba = model.predict_proba(student_df)[0][1]
            st.subheader("Predicted Dropout Probability")
            st.metric(label="", value=f"{proba:.2%}")


        columns= ['Application order',
 'Age at enrollment',
 'Curricular units 1st sem (credited)',
 'Curricular units 1st sem (enrolled)',
 'Curricular units 1st sem (evaluations)',
 'Curricular units 1st sem (approved)',
 'Curricular units 1st sem (grade)',
 'Curricular units 1st sem (without evaluations)',
 'Unemployment rate',
 'Inflation rate',
 'GDP',
 'pass_rate_sem1',
 'economic_stress',
 'not_first_choice',
 'Marital status_Separated',
 'Marital status_Single',
 'Application mode_2nd phase—general contingent',
 'Application mode_3rd phase—general contingent',
 'Application mode_Change in course',
 'Application mode_Change in institution/course',
 'Application mode_Holders of other higher courses',
 'Application mode_Other',
 'Application mode_Over 23 years old',
 'Application mode_Technological specialization diploma holders',
 'Application mode_Transfer',
 'Course_Agronomy',
 'Course_Animation and Multimedia Design',
 'Course_Basic Education',
 'Course_Biofuel Production Technologies',
 'Course_Communication Design',
 'Course_Equiniculture',
 'Course_Informatics Engineering',
 'Course_Journalism and Communication',
 'Course_Management',
 'Course_Management (evening attendance)',
 'Course_Nursing',
 'Course_Oral Hygiene',
 'Course_Social Service',
 'Course_Social Service (evening attendance)',
 'Course_Tourism',
 'Course_Veterinary Nursing',
 'Daytime/evening attendance_Evening',
 'Previous qualification_Higher Education',
 'Previous qualification_Secondary Education',
 'Previous qualification_Technical / Vocational',
 "Mother's qualification_Higher Education",
 "Mother's qualification_No Schooling / Illiterate",
 "Mother's qualification_Secondary Education",
 "Mother's qualification_Technical / Vocational",
 "Father's qualification_Higher Education",
 "Father's qualification_No Schooling / Illiterate",
 "Father's qualification_Other",
 "Father's qualification_Secondary Education",
 "Father's qualification_Technical / Vocational",
 "Mother's occupation_Agriculture",
 "Mother's occupation_Armed Forces",
 "Mother's occupation_Managers",
 "Mother's occupation_Other",
 "Mother's occupation_Professionals",
 "Mother's occupation_Services",
 "Mother's occupation_Skilled manual",
 "Mother's occupation_Technicians",
 "Mother's occupation_Unskilled",
 "Father's occupation_Agriculture",
 "Father's occupation_Armed Forces",
 "Father's occupation_Managers",
 "Father's occupation_Other",
 "Father's occupation_Professionals",
 "Father's occupation_Services",
 "Father's occupation_Skilled manual",
 "Father's occupation_Technicians",
 "Father's occupation_Unskilled",
 'Displaced_Yes',
 'Educational special needs_Yes',
 'Debtor_Yes',
 'Tuition fees up to date_Yes',
 'Gender_Male',
 'Scholarship holder_Yes',
 'International_Yes']

        with open("pipeline9.skops", "rb") as f:
            untrusted = skops.io.get_untrusted_types(data=None, file=f)
            model = sio.load("pipeline9.skops", trusted=untrusted)

        with open("preprocessor9.skops", "rb") as f:
            untrusted = skops.io.get_untrusted_types(data=None, file=f)
            preprocessor = sio.load("preprocessor9.skops", trusted=untrusted)

        with open("X9_train.skops", "rb") as f:
            untrusted = skops.io.get_untrusted_types(data=None, file=f)
            shop_data = sio.load("X9_train.skops", trusted=untrusted)

        student_df = preprocessor.transform(student_df)

        explainer = shap.Explainer(model.predict_proba, shop_data)
        shap_values = explainer(student_df)

        shap_vals_class1 = shap_values.values[:, :, 1] if shap_values.values.ndim == 3 else shap_values.values

        mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)

        student_df = pd.DataFrame(student_df, columns = columns)

        feature_names = student_df.columns
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        })

        shap_importance = shap_importance.sort_values(by='mean_abs_shap', ascending=False)
        values = shap_importance['feature']

        for i, feature in enumerate(values):
            for col in categorical_columns:
                if col in feature:
                    values.iloc[i] = feature.split("_")[0]


        values = values.drop_duplicates().reset_index(drop=True)

        top3 = values[:3]

        st.subheader("Top 3 Features Contributing to the Probability")

        for i, feature in enumerate(top3, start=1):
            importance = shap_importance.loc[shap_importance['feature'] == feature, 'mean_abs_shap'].values[0]
            st.markdown(f"**{i}. {feature}**")
            st.progress(importance / shap_importance['mean_abs_shap'].max())




else:
    st.warning("⚠️ Please fill in all fields to enable Submit.")
