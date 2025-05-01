import streamlit as st
import pandas as pd
import numpy as np
import random

# Set page configuration
st.set_page_config(page_title="Credit Risk Prediction Demo", page_icon="💰", layout="wide")

# App title
st.title("Credit Risk Prediction")
st.write("This is a demonstration of a credit risk prediction app.")

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
        # Simulate prediction based on input values
        # This is a simplified simulation based on some logical rules
        risk_score = 0
        
        # Income affects risk (higher income = lower risk)
        if income > 200000:
            risk_score -= 30
        elif income > 100000:
            risk_score -= 15
        
        # Employment affects risk (longer employment = lower risk)
        if days_employed < -5000:
            risk_score -= 20
        
        # Car and property ownership affect risk
        if own_car == 'Y':
            risk_score -= 10
        if own_realty == 'Y':
            risk_score -= 15
            
        # Children affect risk
        risk_score += children * 5
        
        # Final prediction
        prediction = 'good' if risk_score < 0 else 'bad'
        confidence = min(0.9, max(0.6, abs(risk_score) / 100))
        
        # Show prediction
        st.subheader("Prediction Result")
        
        if prediction == 'good':
            st.success(f"Demo: This client is predicted to be a **GOOD** credit risk with {confidence*100:.1f}% confidence.")
        else:
            st.error(f"Demo: This client is predicted to be a **BAD** credit risk with {confidence*100:.1f}% confidence.")
        
        # Use streamlit's built-in chart
        st.subheader("Prediction Probability")
        probs = {
            'Good': confidence if prediction == 'good' else 1-confidence, 
            'Bad': 1-confidence if prediction == 'good' else confidence
        }
        
        # Convert to DataFrame for better display
        chart_data = pd.DataFrame({
            'Category': ['Good', 'Bad'],
            'Probability': [probs['Good'], probs['Bad']]
        })
        
        st.bar_chart(chart_data.set_index('Category'))

with tab2:
    st.header("Feature Importance")
    st.write("This chart shows simulated feature importance for credit risk prediction.")
    
    # Create a simulated feature importance chart using Streamlit
    features = ['Income', 'Days Employed', 'Property Ownership', 
                'Car Ownership', 'Children', 'Occupation', 
                'Family Status', 'Gender']
    importances = [0.28, 0.23, 0.15, 0.12, 0.10, 0.07, 0.03, 0.02]
    
    # Create DataFrame
    imp_df = pd.DataFrame({
        'Feature': features, 
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Display using Streamlit's bar chart
    st.bar_chart(imp_df.set_index('Feature'))

with tab3:
    st.header("About This App")
    st.write("""
    This application demonstrates a credit risk prediction model based on client application information.
    
    In a production environment, this would use a trained machine learning model, but for demonstration
    purposes, this version uses a simplified approach.
    
    ### Model Information
    - **Features Used**: Gender, Car ownership, Property ownership, Income, Income source, 
      Employment duration, Occupation, Number of children, Family status
    
    ### Interpretation
    - **Good Risk**: Client is likely to pay back loans on time
    - **Bad Risk**: Client might have payment difficulties
    
    ### Limitations
    This is a simplified demonstration model and should not be used for actual credit decisions.
    A production system would use a properly trained machine learning model with more sophisticated
    rules and patterns.
    """)

# Add footer
st.markdown("""
---
*This app is for educational purposes only.*
""")
