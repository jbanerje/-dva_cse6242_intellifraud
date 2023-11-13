payment_type        = ('AA', 'AB', 'AC', 'AD', 'AE')
employment_status   = ('CA', 'CB', 'CC', 'CD', 'CE','CF','CG')
housing_status      = ('BA', 'BB', 'BC', 'BD', 'BE','BF','BG')
device_os           = ('windows','other','linux','macintosh','x11')
customer_age        = ('< 50 yrs', '>= 50 yrs')
classifier_models   = ('LGBMClassifier', 
                       'XGBClassifier', 
                       'AdaBoostClassifier', 
                       'VotingClassifier', 
                       'StackingClassifier')

shap_columns = ['Prev Address (Month)', 
                'Email count', 
                'Credit Score',
                'Age of previous account (Months)',
                'Credit Limit',
                'Customer Age', 
                'Housing Status', 
                'Device OS', 
                'Employment Status',
                'Keep Alive Session', 
                'Has Other Cards', 
                'Has Valid Phone Num',
                'Payment Type'
                ]
