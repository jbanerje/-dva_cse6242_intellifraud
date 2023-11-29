#%%
import os
print(os.getcwd())

#%%
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from datetime import datetime
import config    
from pre_process import *
from feature_importance import *
from explaining.shap_explain import *
import joblib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# All imports here
from sklearn.compose import make_column_selector

from sklearn.metrics import confusion_matrix # Library for model evaluation
from sklearn.metrics import accuracy_score # Library for model evaluation
from sklearn.model_selection import train_test_split # Library to split datset into test and train

from sklearn.dummy import DummyClassifier
from sklearn.linear_model  import LogisticRegression # Logistic Regression Classifier
from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent Classifier
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier
from sklearn.ensemble  import RandomForestClassifier # Random Forest Classifier
from sklearn.neighbors import KNeighborsClassifier # K Nearest neighbors Classifier
from sklearn.naive_bayes import GaussianNB #Naive Bayes Classifier
from sklearn.svm import SVC #Support vector Machine Classifier
from sklearn.ensemble import AdaBoostClassifier # Ada Boost Classifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import plotly.express as px


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


#%%
def display_shap_importance_summary(usr_model):

    explainer, X_train =  get_shap_explainer(usr_model)
    shap_values2 = explainer(X_train[:1])
    # get just the explanations for the positive class
    shap_values = shap_values2[..., 1]

    plt.clf()
    shap_plot = shap.plots.bar(shap_values)
    st.pyplot(plt.gcf())

    return


def display_shap_importance_individual(usr_model, formatted_inp_for_prediction):

    explainer, X_train =  get_shap_explainer(usr_model)
    np_array = np.array([formatted_inp_for_prediction])

    df_formatted_inp_for_prediction = pd.DataFrame(formatted_inp_for_prediction, 
                                                   columns=config.shap_columns)
    #shap_values2 = explainer(X_train[:1])
    shap_values2 = explainer(df_formatted_inp_for_prediction)

    # get just the explanations for the positive class
    shap_values = shap_values2[..., 1]

    plt.clf()
    shap.plots.waterfall(shap_values[0])
    #st.pyplot(plt.gcf())

    return plt.gcf()


def predict_fraud(usr_model, formatted_inp_for_prediction):
    
    ''' Function to predict Fraud/Not Fraud '''
    prediction_map = {0:'Not Fraud', 1:'Fraud'}
    print(usr_model)
    try:
        loaded_model = joblib.load(f'./model/sample_1_1/{usr_model}.pkl')
        prediction     = int(loaded_model.predict(formatted_inp_for_prediction))
        return prediction_map[prediction]
    
    except Exception as e:
        print('Unble to load model:', e)
        return e

def setup_shap_explainer(usr_model):
    
    ''' Function to predict Fraud/Not Fraud '''
    prediction_map = {0:'Not Fraud', 1:'Fraud'}
    print(usr_model)
    try:
        loaded_model = joblib.load(f'./{usr_model}.pkl')
        return loaded_model
    
    except Exception as e:
        print('Unble to load model:', e)
        return e


def display_feature_importance(usr_model):

    feature_imp = calc_feature_importance(usr_model)
    fig = px.bar(feature_imp, x="Feature_Importance", y="Feature", orientation='h')
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    return


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
    #usr_model = st.sidebar.selectbox('Choose Your Model', config.shap_classifier_models)
    


    # Create Input
    with st.form("fraud_detection_form"):
            #customer_age        = st.selectbox("Customer Age", config.customer_age)
            #employment_status   = st.selectbox('Employment Status', config.employment_status)
            housing_status      = st.selectbox('Housing Status', config.housing_status)
            #payment_type        = st.selectbox('Payment Type', config.payment_type)
            
            proposed_credit_limit = st.slider("Credit Limit", 200, 2000, 200)
            device_os           = st.selectbox('Device OS', config.device_os)
    
            phone_home_valid    = st.checkbox("Valid Home Phone")
            has_other_cards     = st.checkbox("Has Other Cards")
            keep_alive_session  = st.checkbox("Keep Alive Session")

            prev_address_months_count   = st.text_input('Prev Address (Month)')
            credit_risk_score           = st.text_input('Credit Score')
            #bank_months_count           = st.text_input('Age of previous account (Months)')
            #date_of_birth_distinct_emails_4w   = st.text_input('Email count with same date of birth in last 4 weeks')
            
            current_address_months_count = st.text_input('Lived number of Monhts at Current Address (# of Months)')

            income = st.text_input('Income')
            name_email_similarity = st.slider(label='Similarity of Email Address to Applicatnt\'s Name', 
                                            min_value=0.1, 
                                            max_value=1.0, 
                                            value=0.5,  # Default value
                                            step=0.01
                                        )


            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")
            if submitted:
                # Format Input
                formatted_inp_for_prediction = map_and_fmt_categorical_column([#customer_age,
                                                                                #employment_status,
                                                                                housing_status,
                                                                                #payment_type,
                                                                                proposed_credit_limit,
                                                                                device_os,
                                                                                phone_home_valid,
                                                                                has_other_cards,
                                                                                keep_alive_session,
                                                                                prev_address_months_count,
                                                                                credit_risk_score,
                                                                                #bank_months_count,
                                                                                #date_of_birth_distinct_emails_4w,
                                                                                current_address_months_count, 
                                                                                income, 
                                                                                name_email_similarity 
                                                                            ])
                # Predict ouput from model
                output = predict_fraud(usr_model, formatted_inp_for_prediction)
                
                if output == 'Fraud':
                    st.error('Its a Fraud Account!', icon="🚨")
                else:
                    st.success('Not a Fraud', icon="✅")

                st.subheader("SHAP Explainability Results of the decision:")                
                #st.subheader("Using SHAP \"Exact\" explainer")

                shap_figure = display_shap_importance_individual(usr_model, formatted_inp_for_prediction)
                st.pyplot(shap_figure)


    
    st.subheader("Model's SHAP Explainability - Global Summary based on the entire dataset:")
    #st.subheader("Using SHAP \"Exact\" explainer")

    display_shap_importance_summary(usr_model)

    
    return

if __name__ == '__main__':
    
    # Streamlit Inteface
    streamlit_interface()
