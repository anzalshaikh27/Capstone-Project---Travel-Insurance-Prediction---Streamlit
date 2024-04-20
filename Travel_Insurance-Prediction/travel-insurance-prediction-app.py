import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV
import plotly.express as px
import plotly.graph_objects as go




# Load and preprocess the dataset
df_url = "https://raw.githubusercontent.com/anzalshaikh27/Capstone-Project---Travel-Insurance-Prediction---Streamlit/main/Travel_Insurance-Prediction/Dataset/TravelInsurancePrediction.csv"
df = pd.read_csv(df_url)
df = df.drop(columns=["Index"], errors='ignore')

# Define categorical columns and convert them to string type
categorical_cols = ["Employment Type", "GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]
df[categorical_cols] = df[categorical_cols].astype(str)

# Initialize and fit OrdinalEncoder with handling unknown categories
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[categorical_cols] = enc.fit_transform(df[categorical_cols])

# Split the dataset
X = df.drop(columns=["TravelInsurance"], errors='ignore')
y = df["TravelInsurance"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the RandomForestClassifier with balanced class weights
model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
model.fit(X_train, y_train)

# Calibrate the model using cross-validation
calibrated_clf = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated_clf.fit(X_train, y_train)

# Store the column order at training time
training_columns = X_train.columns.tolist()

# Streamlit UI

# Page Configuration
st.set_page_config(page_title="Travel Insurance Prediction App", layout="wide",page_icon=":airplane:")

# Define custom styles
st.markdown("""
    <style>
    /* Main content styles */
    .reportview-container .main {
        color: #4f4f4f; /* Darker text for better readability */
        background-color: #fafafa; /* Light background */
    }
    /* Sidebar styles */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        color: #4f4f4f;
        padding-top: 5rem; /* Add padding to align with the main title */
    }
    /* Style for the title */
    h1 {
        color: #2980b9;
    }
    /* Adjust button styles */
    .stButton > button {
        width: 100%;
        border: none;
        background-color: #2980b9;
        color: white;
    }
    /* Other widget styles */
    .st-bd {
        border: 1px solid #efefef;
        border-radius: 5px;
    }
    .st-bb {
        border-bottom: 1px solid #efefef;
    }
    .st-bj {
        background-color: #e8e8e8;
    }
    .Widget>label {
        color: #2980b9;
        font-weight: bold;
    }
    /* Style adjustments for selectbox dropdown */
    .st-dq {
        color: #2980b9;
    }
    .css-2trqyj:focus {
        border-color: #2980b9;
    }
    /* Tooltip adjustments */
    .stTooltip {
        background-color: #2980b9;
    }
    /* Style for the image to fit it within the sidebar */
    .sidebar .sidebar-content img {
        max-width: 100%;
        height: auto;
    }
    </style>
""", unsafe_allow_html=True)



st.title("Travel Insurance Prediction App")
# Banner Image
banner_image_url = "https://raw.githubusercontent.com/anzalshaikh27/Travel-Insurance-Prediction-SL/main/Travel_Insurance-Prediction/banner1.jpeg"

# Display the banner image using HTML and CSS for finer control
html_string = f"""
<div style="text-align: center;">
    <img src="{banner_image_url}" alt="Travel Insurance Banner" style="width: 1050px; height: 300px;"/>
</div>
"""
st.markdown(html_string, unsafe_allow_html=True)
st.markdown(" ")
st.markdown("Provide your details to predict whether you're likely to purchase travel insurance.")
st.markdown(" ")

#gif
logo_gif_url = "https://raw.githubusercontent.com/anzalshaikh27/Travel-Insurance-Prediction-SL/main/Travel_Insurance-Prediction/Riding-Airplane.gif"  


with st.sidebar:
    
    st.header('Traveler Information')
    st.image(logo_gif_url, width=250)
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    employment_type = st.selectbox('Employment Type', options=df['Employment Type'].unique().tolist())
    graduate_or_not = st.selectbox('Graduate Or Not', options=df['GraduateOrNot'].unique().tolist())
    frequent_flyer = st.selectbox('Frequent Flyer', options=df['FrequentFlyer'].unique().tolist())
    ever_travelled_abroad = st.selectbox('Ever Travelled Abroad', options=df['EverTravelledAbroad'].unique().tolist())
    annual_income = st.number_input('Annual Income', min_value=0, value=450000)
    family_members = st.slider('Family Members', 1, 10, 4)
    chronic_diseases = st.selectbox('Chronic Diseases', ['0.0', '1.0'])

submit = st.button('Predict Insurance Need')

if submit:
    
    # Create the input DataFrame for prediction
    input_features = [
        employment_type, graduate_or_not, frequent_flyer,
        ever_travelled_abroad, age, annual_income, family_members,
        chronic_diseases == 'Yes'  
    ]
    input_data = pd.DataFrame([input_features], columns=training_columns)

    # Apply the rule-based logic for 'likely'
    if (age > 30 and annual_income > 500000 and family_members > 4):
        result_text = 'likely'
        prediction_prob = 0.95  
    else:
        # Convert categorical columns to string and encode
        input_data[categorical_cols] = input_data[categorical_cols].astype(str)
        input_data[categorical_cols] = enc.transform(input_data[categorical_cols])

        # Check if all other features are 0.0 for 'not likely'
        if all(input_data[categorical_cols].iloc[0] == 0.0) and age <= 30 and annual_income <= 500000 and family_members <= 4:
            result_text = 'not likely'
            prediction_prob = 0.99  # Assign a high probability for 'not likely'
        else:
            # Predict using the calibrated model
            prediction = calibrated_clf.predict(input_data)
            prediction_prob = calibrated_clf.predict_proba(input_data)[0][prediction[0]]
            result_text = 'likely' if prediction[0] == 1 else 'not likely'

    st.success(f'You are **{result_text}** to purchase travel insurance with a probability of {prediction_prob*100:.2f}%.')


# ROC Curve
if st.checkbox('Show ROC Curve'):
    fpr, tpr, thresholds = roc_curve(y_test, calibrated_clf.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
    roc_fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    roc_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(roc_fig)

# Feature Importance Visualization
if st.checkbox('Show Feature Importance'):
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
    fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance')
    st.plotly_chart(fig)

# Correlation Heatmap
if st.checkbox('Show Correlation Heatmap'):
    correlation = df.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(correlation, text_auto=True, aspect="auto", title='Feature Correlation Heatmap')
    st.plotly_chart(fig)







