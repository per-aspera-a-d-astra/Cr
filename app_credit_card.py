import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
#from sklearn.pipeline import Pipeline
import requests
import os
import pickle
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Credit Risk Prediction App",
    page_icon="💰",
    layout="wide"
)

# Define function to download and load model from Google Drive
@st.cache_resource
def load_model_from_drive():
    file_id = "15MzIDygfeabBBjOBZDsXOs-d4F6vm1bA"
    
    # Try a different download URL format that can handle large files
    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    
    try:
        st.info("Downloading model from Google Drive... This may take a moment.")
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Check if we're getting the confirmation page
        if 'Virus scan warning' in response.text:
            st.warning("Handling Google Drive virus scan warning...")
            # Find the confirmation token
            for k, v in response.cookies.items():
                if k.startswith('download_warning'):
                    # Get the value of the token
                    token = v
                    # Use the token to get the file
                    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
                    response = session.get(url, stream=True)
                    break
        
        # Load the model
        model = pickle.load(BytesIO(response.content))
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Detailed error loading model: {str(e)}")
        st.stop()

# Load the model
model = load_model_from_drive()

# Define the features used in training
features = [
    'CODE_GENDER',          # Gender
    'FLAG_OWN_CAR',         # Car ownership
    'FLAG_OWN_REALTY',      # Property ownership
    'AMT_INCOME_TOTAL',     # Total income
    'NAME_INCOME_TYPE',     # Income source (e.g., employed, retired)
    'DAYS_EMPLOYED',        # Days employed
    'OCCUPATION_TYPE',      # Job type
    'CNT_CHILDREN',         # Number of children
    'NAME_FAMILY_STATUS',   # Marital status
]

# App title
st.title("Credit Risk Prediction")
st.write("This app predicts the credit risk of clients based on their application information.")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Make Prediction", "Feature Importance", "About"])

with tab1:
    st.header("Enter Client Information")
    
    # Create input form for user
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox('Gender', ['M', 'F'])
        own_car = st.selectbox('Owns Car', ['Y', 'N'])
        own_realty = st.selectbox('Owns Property', ['Y', 'N'])
        income = st.number_input('Total Income', min_value=0.0, value=100000.0, step=10000.0)
        income_type = st.selectbox('Income Source', 
                                  ['Working', 'Commercial associate', 'Pensioner', 
                                   'State servant', 'Student'])
    
    with col2:
        days_employed = st.number_input('Days Employed (negative number)', 
                                        min_value=-15000, value=-2000, step=100)
        occupation = st.selectbox('Occupation', 
                                 ['Unknown', 'Laborers', 'Core staff', 'Managers', 
                                  'Drivers', 'Sales staff', 'Accountants', 'Medicine staff',
                                  'Cooking staff', 'Security staff', 'Cleaning staff',
                                  'High skill tech staff', 'Waiters/barmen staff', 'Secretaries',
                                  'Private service staff', 'IT staff', 'Realty agents', 'HR staff'])
        children = st.number_input('Number of Children', min_value=0, value=0, step=1)
        family_status = st.selectbox('Family Status', 
                                    ['Married', 'Single / not married', 'Civil marriage', 
                                     'Separated', 'Widow'])
    
    # Prediction button
    predict_button = st.button('Predict Credit Risk')
    
    if predict_button:
        # Create a dataframe with the input values
        input_data = pd.DataFrame({
            'CODE_GENDER': [gender],
            'FLAG_OWN_CAR': [own_car],
            'FLAG_OWN_REALTY': [own_realty],
            'AMT_INCOME_TOTAL': [income],
            'NAME_INCOME_TYPE': [income_type],
            'DAYS_EMPLOYED': [days_employed],
            'OCCUPATION_TYPE': [occupation],
            'CNT_CHILDREN': [children],
            'NAME_FAMILY_STATUS': [family_status],
        })
        
        # Make prediction
        try:
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
            
            # Show prediction
            st.subheader("Prediction Result")
            
            if prediction[0] == 'good':
                st.success(f"This client is predicted to be a **GOOD** credit risk with {prediction_proba[0][0]*100:.1f}% confidence.")
            else:
                st.error(f"This client is predicted to be a **BAD** credit risk with {prediction_proba[0][1]*100:.1f}% confidence.")
            
            # Show probability chart
            st.subheader("Prediction Probability")
            prob_data = pd.DataFrame({
                'Category': ['Good', 'Bad'],
                'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Category', y='Probability', data=prob_data, ax=ax)
            ax.set_ylim(0, 1)
            for i, bar in enumerate(ax.patches):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{prob_data.Probability.iloc[i]:.1%}',
                    ha='center', va='bottom', fontsize=12
                )
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error making prediction: {e}")

with tab2:
    st.header("Feature Importance")
    st.write("This chart shows which features are most important for predicting credit risk.")
    
    # Get feature importance
    try:
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importances = model.named_steps['classifier'].feature_importances_
        
        # Create a dataframe
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), ax=ax)
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Could not extract feature importance: {e}")

with tab3:
    st.header("About This App")
    st.write("""
    This application uses a machine learning model to predict credit risk based on client application information.
    
    The model was trained on historical credit data and evaluates whether a client is likely to have good or bad credit behavior.
    
    ### Model Information
    - **Model Type**: Random Forest Classifier
    - **Features Used**: Gender, Car ownership, Property ownership, Income, Income source, 
      Employment duration, Occupation, Number of children, Family status
    - **Accuracy**: Approximately 70-80% (may vary)
    
    ### Interpretation
    - **Good Risk**: Client is likely to pay back loans on time
    - **Bad Risk**: Client might have payment difficulties
    
    ### Limitations
    This is a prediction model and should be used as one of many factors in credit decisions.
    The model has been trained on historical data and may not capture all real-world scenarios.
    """)

# Add footer
st.markdown("""
---
*This app is for educational purposes only.*
""")
