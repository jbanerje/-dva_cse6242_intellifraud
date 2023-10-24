import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd


''' Visualization code to build interactive dashboard for Intellifraud tool'''
# Load in dataset
 #df = pd.read_csv("../Data/Base.csv")
df = pd.read_csv("subset_viz_data.csv")
# Create a Dash web application
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Intellifraud: An Early Fraud Detection Tool"),

    # Dropdown to select a feature for visualization
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns],
        value='income'  # Default selection
    ),

    # Scatter plot to visualize selected feature
    dcc.Graph(id='scatter-plot'),

    # Bar chart for distribution of fraud
    dcc.Graph(id='fraud-distribution'),
])


# Callback to update the scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_scatter_plot(selected_feature):
    fig = px.scatter(df, x=selected_feature, y='credit_risk_score', color='fraud_bool',
                     title=f'Scatter Plot for {selected_feature}')
    return fig

# Callback to update the bar chart for fraud distribution
@app.callback(
    Output('fraud-distribution', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_fraud_distribution(selected_feature):
    # Create a bar chart to show the distribution of fraud based on the selected feature
    fraud_distribution = df.groupby(selected_feature)['fraud_bool'].mean().reset_index()
    fig = px.bar(fraud_distribution, x=selected_feature, y='fraud_bool',
                 title=f'Fraud Distribution by {selected_feature}')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
