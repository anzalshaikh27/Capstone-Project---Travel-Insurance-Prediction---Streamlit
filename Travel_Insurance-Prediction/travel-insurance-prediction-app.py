import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from fpdf import FPDF

# Set page config
st.set_page_config(page_title='Travel Insurance Prediction App', page_icon=':airplane:')

# Custom CSS for the sidebar
sidebar_style = """
<style>
[data-testid="stSidebar"] > div:first-child {
    background-image: linear-gradient(rgba(0, 123, 255, 0.6), rgba(0, 123, 255, 0.2)), url('https://example.com/travel-insurance-background.jpg');
    background-size: cover;
    color: white;
}
[data-testid="stSidebar"] .css-1lcbmhc {
    background-color: rgba(255, 255, 255, 0);
}
</style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

# Load and preprocess the dataset
df = pd.read_csv("https://raw.githubusercontent.com/anzalshaikh27/Capstone-Project---Travel-Insurance-Prediction---Streamlit/main/Travel_Insurance-Prediction/Dataset/TravelInsurancePrediction.csv")
enc = OrdinalEncoder()
df[["Employment Type", "GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]] = enc.fit_transform(df[["Employment Type", "GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]])

# Define Independent and Dependent Variables
X = df.drop("TravelInsurance", axis=1)
y = df["TravelInsurance"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Sidebar for user input fields
with st.sidebar:
    st.header('Traveler Information')
    age = st.slider('Age', 18, 100, 30)
    employment_type = st.selectbox('Employment Type', df['Employment Type'].unique())
    graduate_or_not = st.selectbox('Graduate Or Not', ['Yes', 'No'])
    annual_income = st.number_input('Annual Income', value=500000)
    family_members = st.slider('Family Members', 1, 10, 4)
    chronic_diseases = st.selectbox('Chronic Diseases', ['No', 'Yes'])
    frequent_flyer = st.selectbox('Frequent Flyer', ['No', 'Yes'])
    ever_travelled_abroad = st.selectbox('Ever Travelled Abroad', ['No', 'Yes'])
    submit = st.button('Predict Insurance Need')

# Main content
st.write("# Travel Insurance Prediction App")
st.markdown("""
This app predicts the likelihood of a traveler purchasing travel insurance based on their profile.
Fill out the traveler information in the sidebar and press 'Predict Insurance Need' to see the prediction.
""")

if submit:
    # Preprocess inputs
    input_df = pd.DataFrame({
        'Age': [age],
        'Employment Type': [employment_type],
        'GraduateOrNot': [graduate_or_not],
        'AnnualIncome': [annual_income],
        'FamilyMembers': [family_members],
        'ChronicDiseases': [1 if chronic_diseases == 'Yes' else 0],
        'FrequentFlyer': [1 if frequent_flyer == 'Yes' else 0],
        'EverTravelledAbroad': [1 if ever_travelled_abroad == 'Yes' else 0]
    })

    # Encode categorical variables
    input_df[["Employment Type", "GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]] = enc.transform(input_df[["Employment Type", "GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]])

    # Make prediction
    prediction = model.predict(input_df)
    prediction_prob = model.predict_proba(input_df)

    # Display result
    if prediction[0] == 0:
        st.success(f'The model predicts: **Not likely** to purchase travel insurance with a probability of {prediction_prob[0][0]*100:.2f}%.')
    else:
        st.success(f'The model predicts: **Likely** to purchase travel insurance with a probability of {prediction_prob[0][1]*100:.2f}%.')

# Note: Further customizations and functionalities such as PDF report generation can be added similarly to the original example.

    # Create the PDF
    pdf_content = create_pdf_report(
        gender=gender,
        age=age,
        hypertension=hypertension,
        heart_disease=heart_disease,
        smoking_history=smoking_history,
        bmi=bmi,
        hba1c_level=hba1c_level,
        blood_glucose_level=blood_glucose_level,
        prediction=prediction
    )

    # Convert to a bytes object
    pdf_bytes = bytes(pdf_content)

    # Use Streamlit's download button to offer the PDF to the user
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="diabetes_prediction_report.pdf",
        mime="application/pdf",
    )

    # Feature importance graph and summary inside a toggle
    with st.expander("View Feature Importance"):
        feature_names = X_train.columns
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
    
    fig = px.bar(importance_df, x='feature', y='importance', title='Feature Importance')
    fig.update_layout(xaxis_title='Feature', yaxis_title='Importance')
    st.plotly_chart(fig)
    
    # Summary of feature importances
    top_features = importance_df['feature'].iloc[:2].tolist()
    st.markdown(f"The most influential factors in predicting diabetes are **{top_features[0]}** and **{top_features[1]}**.")

    
#Install plotly, fpdf and streamlit-extras for code to run
