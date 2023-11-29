
#%%
import joblib
import xgboost
import numpy as np
import pandas as pd
import os
import shap
import sys
from pathlib import Path


#%%

def load_data():

    script_directory = os.path.dirname(os.path.abspath(__file__))
    #print("script directory:", script_directory)

    #current_working_directory = os.getcwd()
    #print("current_working_directory", current_working_directory)

    load_directory = script_directory + "/../data/Base.csv"

    input_df = pd.read_csv(load_directory)
    #print(f'Rows - {input_df.shape[0]}, Columns - {input_df.shape[1]}')
    #input_df.head()

    # Extract Continuous & Categorical Columns
    cat_cols = input_df.select_dtypes(include=['object']).columns.tolist()
    cont_cols = input_df.select_dtypes(exclude=['object']).columns.tolist()

    return input_df, cat_cols, cont_cols

#%%
def map_categorical_column(df):
    
    ''' Function to map the categorical columns '''
     
    map_payment_type      = {'AA':0, 'AB':1, 'AC':2, 'AD':3, 'AE':4}
    map_employment_status = {'CA':0, 'CB':1, 'CC':2, 'CD':3, 'CE':4,'CF':5,'CG':6}
    map_housing_status    = {'BA':0, 'BB':1, 'BC':2, 'BD':3, 'BE':4,'BF':5,'BG':6}
    map_source            = {'INTERNET':0,'TELEAPP':1}
    map_device_os         = {'windows':0,'other':1,'linux':2,'macintosh':3,'x11':4}
    
    # Updating the mapping in dataframe
    df["payment_type"]                 = df["payment_type"].map(map_payment_type)
    df["employment_status"]            = df["employment_status"].map(map_employment_status)
    df["housing_status"]               = df["housing_status"].map(map_housing_status)
    df["source"]                       = df["source"].map(map_source)
    df["device_os"]                    = df["device_os"].map(map_device_os)

    return df
# %%

#input_df_copy = input_df.copy()
#input_df_num = map_categorical_column(input_df_copy)
#input_df_num.head()




# %%

def load_model(usr_model):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    model_location = f'../model/sample_1_1/{usr_model}.pkl' 
    load_location = script_directory + "/" + model_location
    #print("Model's load_directory: ", load_location)
    print(usr_model)
    try:

        loaded_model = joblib.load(load_location)
        return loaded_model
    
    except Exception as e:
        print('Unble to load model:', e)
        return e

# %%
# Note: Function copied from: /training/intellifraud-data-modelling.ipynb

def create_sample_set(train_df, non_fraud_sample_sizse):
    
    
    # Select columns
    train_df = train_df[[ 
                        #'date_of_birth_distinct_emails_4w', #--old                         
                        #'bank_months_count', #--old
                        #'customer_age', #--old
                        #'employment_status', #--old
                        #'payment_type', #--old 
                        'housing_status', 
                        'device_os', 
                        'credit_risk_score',
                        'current_address_months_count', #new 
                        'has_other_cards', 
                        'keep_alive_session', 
                        'prev_address_months_count', 
                        'phone_home_valid', 
                        'proposed_credit_limit', 
                        'name_email_similarity', #new 
                        'income', #new
                        'fraud_bool', 
                        'month']]
    
    #train_df = train_df[['prev_address_months_count', 'date_of_birth_distinct_emails_4w','credit_risk_score', 'bank_months_count', 
    #                     'proposed_credit_limit','customer_age', 'housing_status','device_os', 'employment_status',
    #                     'keep_alive_session','has_other_cards','phone_home_valid','payment_type', 'fraud_bool', 'month']]


    # Fraud Transactions
    train_df_fraud = train_df[train_df.fraud_bool == 1]
    #display(f'Shape of train_df_fraud {train_df_fraud.shape}')
    
    # Non Fraud Transactions
    train_df_non_fraud = train_df[train_df.fraud_bool == 0].sample(train_df_fraud.shape[0] * non_fraud_sample_sizse)
    #display(f'Shape of train_df_non_fraud {train_df_non_fraud.shape}')
    
    # Merge Fraud & Non Fraud
    train_df_merged = pd.concat([train_df_fraud, train_df_non_fraud])

    # Shuffle
    train_df_merged.iloc[:,:] = train_df_merged.sample(frac=1,random_state=123,ignore_index=True)
    #display(f'After merge & shuffle Shape of train_df_merged {train_df_merged.shape}')
#     display(f'Value Counts:\n {train_df_merged["fraud_bool"].value_counts(normalize=True)}')
    
    # X & Y
    X                 = train_df_merged.drop(columns=['fraud_bool'])
    #X['customer_age'] = X['customer_age'].apply(lambda x: 0 if x < 50 else 1)
    y                 = train_df_merged[['fraud_bool', 'month']]
    
    # Train Dataframe
    X_train = X[X.month <= 6].drop(columns=['month'])
    y_train = y[y.month <= 6].drop(columns=['month']).values.ravel()

    # Test Dataframe
    X_test = X[X.month > 6].drop(columns=['month'])
    y_test = y[y.month > 6].drop(columns=['month']).values.ravel()

    return X_train, y_train, X_test, y_test



def create_shap_explainer(model, X_train, y_train):

    clustering = shap.utils.hclust(X_train, y_train)
    masker = shap.maskers.Partition(X_train, clustering=clustering)

    # build an Exact explainer and explain the model predictions on the given dataset
    explainer = shap.explainers.Exact(model.predict_proba, masker)
    #explainer = shap.explainers.Exact(model, masker)

    return explainer


#%%
# TODO: Allow changing "XGBClassifier" to other models dynamically.
def get_shap_explainer(usr_model):

    input_df, cat_cols, cont_cols = load_data()
    input_df_copy = input_df.copy()
    input_df_num = map_categorical_column(input_df_copy)
    X_train, y_train, X_test, y_test = create_sample_set(input_df_num, 1)

    if usr_model == 'LGBMClassifier': 
            model = load_model("LGBMClassifier")
            explainer = create_shap_explainer(model, X_train, y_train)

    elif usr_model == 'XGBClassifier': 
            model = load_model("XGBClassifier")
            explainer = create_shap_explainer(model, X_train, y_train)

    elif usr_model == 'AdaBoostClassifier':
            model = load_model("LGBMClassifier")
            explainer = create_shap_explainer(model, X_train, y_train)

    elif usr_model == 'VotingClassifier':
            model = load_model("XGBClassifier")
            explainer = create_shap_explainer(model, X_train, y_train)
            #explainer = shap.KernelExplainer(model.predict_proba, X_train)        

    elif usr_model == 'StackingClassifier':
            model = load_model("LGBMClassifier")
            explainer = create_shap_explainer(model, X_train, y_train)
            #explainer = shap.KernelExplainer(model.predict_proba, X_train)     
    else: 
        explainer = None

    return explainer, X_train

#%%

def run_shap_explainer():
    explainer, X_train =  get_shap_explainer()
    #%%
    shap_values2 = explainer(X_train[:1])
    # get just the explanations for the positive class
    shap_values2 = shap_values2[..., 1]
    # %%
    shap.plots.bar(shap_values2)
    # %%
    shap.plots.waterfall(shap_values2[0])
    return
