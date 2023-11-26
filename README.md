# Project
IntelliFraud: Bank Account Fraud Detection

## Goal
Goal of this application is to detect fraudulent bank account getting opened through online applications in a consumer bank. 
 - Our End to End application performs EDA, Looks at various model performances, Creates a graph network to detect fraud transaction flow and Creates an inferencing with feature importance and explanation of prediction thru SHAP.
 - This project demonstrates the use of graph network to analyze any fraudelent transaction pattern in the data.
 - Additionally, this project explores the possibility of getting better predictions thru voting and stacking classifiers comparing with LightGBM, XGBoost and AdaBoost.

## Data 
For our project, we referenced the Kaggle competition  Bank Account Fraud Dataset Suite (NeurIPS 2022)  [ https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 ] that provided 6 datasets which are highly imbalaced.

![alt text](images/data.jpg)

## Artifacts
Our application is desgined using Streamlit Pages concept.
1. Upon launching `app.py`, it launches the home screen and on left we have 4 different pages to select from. 

2. All pages are loaded from `/pages` directory.
    * 1_EDA.py - Performs EDA on data and provides users interactivity.
    * 2_Fraud_Network_Analysis.py - Vizualiton of Graph network showing the fraudulent transaction flow thru various networks and subsystems.
    * 3_Modelling.py - Explores LightGBM, XGBoost and AdaBoost, Voting and Stacking Classifier models and performances.
    * 4_Inference.py - Creates inference upon user input and explains the reasoning for detecting fraud or not.

3. We have 4 notebooks in `/notebooks` directory (Not Required to launch the application.) These have been created a step by step process to build our application.
    * exploratory_data_analysis.ipynb - Notebook for basic EDA and Exploration for modelling.
    * network_visualization.ipynb - Vizualiton of Graph network showing the fraudulent transaction flow thru various networks and subsystems.
    * intellifraud-data-modelling-all-var.ipynb - Building the models.

## Installation
1. Clone the repository from github.
2. Create a virtual environment called intellifraud -  [ https://code.visualstudio.com/docs/python/environments ]
3. Install all required packages from requirements.txt - [`pip install -r requirements.txt`]
4. Create a folder called "data" at the same directory level as app.py. Open terminal and type `mkdir data`
5. Copy the following datasets from Kaggle:
    /kaggle/input/bank-account-fraud-dataset-neurips-2022/Base.csv
    /kaggle/input/bank-account-fraud-dataset-neurips-2022/Variant I.csv
    /kaggle/input/bank-account-fraud-dataset-neurips-2022/Variant II.csv
    /kaggle/input/bank-account-fraud-dataset-neurips-2022/Variant III.csv
    /kaggle/input/bank-account-fraud-dataset-neurips-2022/Variant IV.csv
    /kaggle/input/bank-account-fraud-dataset-neurips-2022/Variant V.csv

## Execution
1. Without docker - `streamlit run app.py` . Application will open at localhost:8501 automatically.
2. With docker, run the following commands to create the image/container and run from docker
    * `docker build --no-cache -t intellifraud` .
    * `docker run -p 8082:8501 intellifraud`
    * On the browser type `localhost:8082`

