
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Sample data (replace with your actual data)
data = pd.DataFrame({
    'income': [0.6, 0.8, 0.5, 0.9, 0.7],
    'customer_age': [30, 40, 25, 35, 28],
    'zip_count_4w': [1500, 2000, 1000, 2500, 1800],
    'credit_risk_score': [200, 300, 180, 250, 220],
    'fraud_bool': [0, 1, 0, 1, 0]
})

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),

    dcc.Dropdown(
        id='data-filter',
        options=[
            {'label': 'All', 'value': 'all'},
            {'label': 'Fraudulent', 'value': 'fraudulent'},
            {'label': 'Non-Fraudulent', 'value': 'non-fraudulent'}
        ],
        value='all',
        multi=False
    ),

    dcc.Graph(id='income-vs-age-scatter'),

    dcc.Graph(id='credit-score-histogram')
])


# Define callbacks to update the graphs based on user input
@app.callback(
    [Output('income-vs-age-scatter', 'figure'),
     Output('credit-score-histogram', 'figure')],
    [Input('data-filter', 'value')]
)
def update_graphs(selected_filter):
    if selected_filter == 'all':
        filtered_data = data
    elif selected_filter == 'fraudulent':
        filtered_data = data[data['fraud_bool'] == 1]
    else:
        filtered_data = data[data['fraud_bool'] == 0]

    # Create a scatter plot for income vs. age
    scatter_fig = px.scatter(
        filtered_data, x='income', y='customer_age',
        color='fraud_bool', labels={'fraud_bool': 'Fraudulent'},
        title='Income vs. Customer Age'
    )

    # Create a histogram for credit scores
    histogram_fig = px.histogram(
        filtered_data, x='credit_risk_score',
        color='fraud_bool', labels={'fraud_bool': 'Fraudulent'},
        title='Credit Risk Score Distribution'
    )

    return scatter_fig, histogram_fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
