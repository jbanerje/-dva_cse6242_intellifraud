
import streamlit as st
from PIL import Image
import pandas as pd
import config        
from pre_process import *
import joblib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import shap
shap.initjs()

def predict_fraud(usr_model, formatted_inp_for_prediction):
    
    ''' Function to predict Fraud/Not Fraud '''
    
    prediction_map = {0:'Not Fraud', 1:'Fraud'}
    print(usr_model)
    
    try:
        loaded_model = joblib.load(f'./model/sample_1_1/{usr_model}.pkl')
        prediction     = int(loaded_model.predict(formatted_inp_for_prediction))
        
        return prediction_map[prediction], loaded_model
    
    except Exception as e:
        print('Unble to load model:', e)
        return e

def streamlit_interface():
    
    # Page Setup
    st.set_page_config(
                        page_title="IntelliFraud",
                        layout="centered",
                        initial_sidebar_state="auto",
                        page_icon='./images/fraud-detection.png',
                    )
    # Hide Footer Setup
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Condense Layout
    padding = 0
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {padding}rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """, unsafe_allow_html=True)

    # Header
    image = Image.open("./images/intellifraud_icon.png")
    st.image(image, width = 300)
    
    # Print the Header Banner
    html_str = f"""
                    <h3 style="background-color:#00A1DE; text-align:center; font-family:arial;color:white">FRAUD DETECTION SYSTEM </h3>
                """

    st.markdown(html_str, unsafe_allow_html=True)
    st.divider()
    
    # Select Model
    usr_model = st.sidebar.selectbox('Choose Your Model', config.classifier_models)
    
    # Create Input
    with st.form("fraud_detection_form"):

            housing_status          = st.selectbox('Housing Status', config.housing_status) #
            device_os               = st.selectbox('Device OS', config.device_os) #
            
            proposed_credit_limit   = st.slider("Credit Limit", 200, 2000, 200) #
            income                  = st.slider("Income", 0.0, 1.0, 0.01) #
            
            has_other_cards         = st.checkbox("Has Other Cards") #
            keep_alive_session      = st.checkbox("Keep Alive Session") #
            phone_home_valid        = st.checkbox("Valid Home Phone") #
            
            name_email_similarity           = st.text_input('Name Email Similarity') #
            current_address_months_count    = st.text_input('Current Address (Month)') #
            prev_address_months_count       = st.text_input('Previous Address (Month)') #
            credit_risk_score               = st.text_input('Credit Score') #
            
            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                
                # Format Input
                formatted_inp_for_prediction = map_and_fmt_categorical_column([
                                                                                housing_status,
                                                                                device_os,
                                                                                proposed_credit_limit,
                                                                                income,
                                                                                has_other_cards,
                                                                                keep_alive_session,
                                                                                phone_home_valid,
                                                                                name_email_similarity,
                                                                                current_address_months_count,
                                                                                prev_address_months_count,
                                                                                credit_risk_score
                                                                            ])

                # Predict ouput from model
                output, classifier_model = predict_fraud(usr_model, formatted_inp_for_prediction)
                
                if output == 'Fraud':
                    st.error('Its a Fraud Account!', icon="🚨")
                else:
                    st.success('Not a Fraud', icon="✅")

                # # Plot Shap Figures
                # explainer = shap.Explainer(classifier_model)
                # st.pyplot(shap.plots.force(explainer.expected_value[0],
                #                            shap_values[0][0]
                #                            formatted_inp_for_prediction,
                #                            feature_names = config.reqd_col_modelling[:-1],
                #                            matplotlib = True
                #                            )
                #         )
    return

if __name__ == '__main__':
    
    # Streamlit Inteface
    streamlit_interface()

    
    