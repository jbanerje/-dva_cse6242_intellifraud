import pandas as pd
import numpy as np

def map_and_fmt_categorical_column(user_input):
    
    ''' Function to map the categorical columns '''
     
    map_payment_type      = {'AA':0, 'AB':1, 'AC':2, 'AD':3, 'AE':4}
    map_employment_status = {'CA':0, 'CB':1, 'CC':2, 'CD':3, 'CE':4,'CF':5,'CG':6}
    map_housing_status    = {'BA':0, 'BB':1, 'BC':2, 'BD':3, 'BE':4,'BF':5,'BG':6}
    map_source            = {'INTERNET':0,'TELEAPP':1}
    map_device_os         = {'windows':0,'other':1,'linux':2,'macintosh':3,'x11':4}
    map_cust_age          = {'< 50 yrs':0, '>= 50 yrs':1}
    
    num_cust_age            = map_cust_age[user_input[0]]
    num_employment_status   = map_employment_status[user_input[1]]
    num_housing_status      = map_housing_status[user_input[2]]
    num_payment_type        = map_payment_type[user_input[3]]
    num_credit_limit        = int(user_input[4])
    num_device_os           = map_device_os[user_input[5]]
    num_valid_phone         = int(user_input[6])
    num_has_oth_cards       = int(user_input[7])
    num_session             = int(user_input[8])
    num_prev_add            = int(user_input[9])
    num_credit_score        = int(user_input[10])
    num_age_prev_accnt      = int(user_input[11])
    num_email_cnt           = int(user_input[12])

    streamlit_form_layout = [num_cust_age, 
                                num_employment_status, 
                                num_housing_status,
                                num_payment_type,
                                num_credit_limit,
                                num_device_os,
                                num_valid_phone,
                                num_has_oth_cards,
                                num_session,
                                num_prev_add,
                                num_credit_score,
                                num_age_prev_accnt,
                                num_email_cnt
                        ]
    
    return np.array([num_prev_add,
                        num_email_cnt,
                        num_credit_score,
                        num_age_prev_accnt,
                        num_credit_limit,
                        num_cust_age,
                        num_housing_status,
                        num_device_os,
                        num_employment_status,
                        num_session,
                        num_has_oth_cards,
                        num_valid_phone,
                        num_payment_type
                ]).reshape(1, -1)

