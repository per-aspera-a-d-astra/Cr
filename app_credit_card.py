import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        # Instead of an actual prediction, show a demo result
        import random
        result = random.choice(['good', 'bad'])
        prob = random.uniform(0.6, 0.9)
        
        if result == 'good':
            st.success(f"Demo: This client is predicted to be a **GOOD** credit risk with {prob*100:.1f}% confidence.")
        else:
            st.error(f"Demo: This client is predicted to be a **BAD** credit risk with {prob*100:.1f}% confidence.")
        
        # Show a demo chart
        try:
            fig, ax = plt.subplots()
            data = {'good': prob if result == 'good' else 1-prob, 'bad': 1-prob if result == 'good' else prob}
            ax = sns.barplot(x=list(data.keys()), y=list(data.values()))
            for i, v in enumerate(data.values()):
                ax.text(i, v + 0.01, f'{v:.1%}', ha='center')
            ax.set_ylim(0, 1)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not display chart: {e}")

with tab2:
    st.header("Feature Importance")
    st.write("This chart shows simulated feature importance for credit risk prediction.")
    
    try:
        # Create a simulated feature importance chart
        features = ['Income', 'Days Employed', 'Property Ownership', 
                    'Car Ownership', 'Children', 'Occupation', 
                    'Family Status', 'Gender']
        importances = [0.28, 0.23, 0.15, 0.12, 0.10, 0.07, 0.03, 0.02]
        
        imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        imp_df = imp_df.sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=imp_df, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display feature importance: {e}")

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
