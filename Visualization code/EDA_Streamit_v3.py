import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import math
import cProfile
import pstats
from io import StringIO

# # # MUST HAVE TO FORCE WIDE MODE! DO NOT MOVE!!! # # #
# Set the page layout
st.set_page_config(
    layout="wide",  # Use the wide layout to take full advantage of the screen width
    initial_sidebar_state="auto",  # Auto-hide the sidebar
)

# Define a fixed size for the plots
plot_width = 700
plot_height = 700

# Load dataset
data = pd.read_csv('../Data/Base.csv')  # Replace with your dataset file

# Replace column names with easier to read names
column_labels = {
    'fraud_bool': 'Fraud Detection (Binary)',
    'income': 'Income',
    'name_email_similarity': 'Name-Email Similarity',
    'prev_address_months_count': 'Previous Address Months Count',
    'current_address_months_count': 'Current Address Months Count',
    'customer_age': 'Customer Age',
    'days_since_request': 'Days Since Request',
    'intended_balcon_amount': 'Intended Balcon Amount',
    'payment_type': 'Payment Type',
    'zip_count_4w': 'Zip Code Count (Last 4 Weeks)',
    'velocity_6h': 'Velocity (Last 6 Hours)',
    'velocity_24h': 'Velocity (Last 24 Hours)',
    'velocity_4w': 'Velocity (Last 4 Weeks)',
    'bank_branch_count_8w': 'Bank Branch Count (Last 8 Weeks)',
    'date_of_birth_distinct_emails_4w': 'Date of Birth Distinct Emails (Last 4 Weeks)',
    'employment_status': 'Employment Status',
    'credit_risk_score': 'Credit Risk Score',
    'email_is_free': 'Email is Free',
    'housing_status': 'Housing Status',
    'phone_home_valid': 'Phone (Home) Valid',
    'phone_mobile_valid': 'Phone (Mobile) Valid',
    'bank_months_count': 'Bank Months Count',
    'has_other_cards': 'Has Other Cards',
    'proposed_credit_limit': 'Proposed Credit Limit',
    'foreign_request': 'Foreign Request',
    'source': 'Source',
    'session_length_in_minutes': 'Session Length (Minutes)',
    'device_os': 'Device OS',
    'keep_alive_session': 'Keep Alive Session',
    'device_distinct_emails_8w': 'Device Distinct Emails (Last 8 Weeks)',
    'device_fraud_count': 'Device Fraud Count',
    'month': 'Month'
}

# Adding discrete columns
discrete_columns = [
    'fraud_bool',
    'income',
    'customer_age',
    'email_is_free',
    'phone_home_valid',
    'phone_mobile_valid',
    'has_other_cards',
    'proposed_credit_limit',
    'foreign_request',
    'keep_alive_session',
    'device_distinct_emails_8w',
    'device_fraud_count',
    'month',
    # Add more columns as needed
]


# Define color-blind friendly palettes
colorblind_palettes = {
    "Palette 1": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"],
    "Palette 2": ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00"],
    "Palette 3": ["#f6c141", "#bda86e", "#8a9e64", "#dcb0ff", "#ff9da6", "#ff82a9", "#0f4b6e", "#bfbdc1", "#ffaec4"],
    "Palette 4": ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"],
    "Palette 5": ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#87CEEB", "#FFA07A", "#F6546A"],
    "Palette 6": ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494"],
    "Palette 7": ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"],
    "Palette 8": ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3", "#8dd3c7"],
    "Palette 9": ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"],
    "Palette 10": ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666", "#e7298a"],
}


st.set_option('deprecation.showPyplotGlobalUse', False)

# Add Streamlit code to create the interactive dashboard
st.markdown(
    """
    <style>
    .stApp {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Add Streamlit code to create the interactive dashboard
st.title('Exploratory Data Analysis')

# Add widgets (e.g., sliders, dropdowns, etc.) for user interaction
selected_feature = st.selectbox('Select a feature for analysis', list(column_labels.values()))

# Display basic statistics
selected_column = [col for col, label in column_labels.items() if label == selected_feature][0]

# Columns for layout of dashboard
col1, col2 = st.columns(2)


# Check if the selected column is numeric before creating the plots
if data[selected_column].dtype in ['int64', 'float64']:
    # Numeric data
    is_discrete = selected_column in discrete_columns

    # Display interactive count plot for discrete columns
    if is_discrete:
        fig = px.histogram(data, x=selected_column, color=selected_column, barmode='overlay',
                           category_orders={selected_column: sorted(data[selected_column].unique())})
        fig.update_layout(
            title=f"Count Plot of {selected_feature}",
            xaxis_title=selected_feature,
            title_x=0.5,
            title_y=0.92,
            yaxis_title="Count",
            width=800,
            height=plot_height,
        )
        st.plotly_chart(fig)
      #  col2.plotly_chart(fig)
    else:
        # Display violin plot for non-binary numeric columns
        fig_violin = go.Figure()
        fig_violin.add_trace(go.Violin(
            y=data[selected_column],
            box_visible=True,
            line_color="#000000",
            fillcolor="#FFFFFF",
            opacity=0.6
        ))
        fig_violin.update_layout(
            title=f"{selected_feature} Distribution",
            title_x=0.5,
            title_y=0.9,
            title_xanchor="center",
            title_yanchor="top",
            xaxis_title=selected_feature,
            yaxis_title="Density",
            width=800,
            height=plot_height,
        )
        col1.plotly_chart(fig_violin)

        # Histogram for non-binary numeric columns
        hist_fig = px.histogram(data, x=selected_column, nbins=30, color_discrete_sequence=['lightblue'])
        hist_fig.update_layout(
            title=f"{selected_feature} Histogram",
            title_x=0.5,
            title_y=0.92,
            title_xanchor="center",
            title_yanchor="top",
            xaxis_title=selected_feature,
            yaxis_title="Frequency",
            hovermode="closest",
            width=800,
            height=plot_height,
            hoverlabel=dict(bgcolor="white", bordercolor="gray"),
        )
        hist_fig.update_traces(
            hoverinfo="x+y+name",
            name=f"Mean: {data[selected_column].mean():.2f}<br>Median: {data[selected_column].median():.2f}",
            hoverlabel=dict(bgcolor="#282828", font=dict(color="white"))
        )
        col2.plotly_chart(hist_fig)

        # Remove the third container
        #col3.empty()

else:
    # Categorical data (including discrete data)
    st.write("This is a non-numeric column. Displaying interactive count plot instead:")
    st.dataframe(data[selected_column].value_counts())
    st.bar_chart(data[selected_column].value_counts())




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if data[selected_column].dtype in ['int64', 'float64'] and len(data[selected_column].unique()) > 2:
    # Check if the feature is non-binary and numeric
    # Separate data for Not Fraud and Fraud
    not_fraud_data = data[data['fraud_bool'] == 0][selected_column]
    fraud_data = data[data['fraud_bool'] == 1][selected_column]

    # Create a Plotly figure
    fig = go.Figure()

    # Create kernel density plot for Not Fraud
    fig.add_trace(go.Histogram(
        x=not_fraud_data,
        histnorm='probability density',
        marker_color='#8FC4C0',  # Set color for Not Fraud
        opacity=1,          # Adjust opacity for transparency
        name='Not Fraud'
    ))

    # Create kernel density plot for Fraud
    fig.add_trace(go.Histogram(
        x=fraud_data,
        histnorm='probability density',
        marker_color='#F0B3BE',   # Set color for Fraud
        opacity=1,          # Adjust opacity for transparency
        name='Fraud'
    ))

    # Update layout and labels
    fig.update_layout(
        title=f"Kernel Density Plot of {selected_feature}",
        xaxis_title=selected_feature,
        yaxis_title="Density",
        title_x=0.50,  # Center the title horizontally
        title_y=0.92,  # Position the title closer to the top
        title_xanchor="center",  # Center the title horizontally
        title_yanchor="top",  # Position the title at the top
        width=1200,  # Adjust the width of the figure
        height=plot_height,  # Adjust the height of the figure
    )


    # Show the interactive plot
    st.plotly_chart(fig)
else:
    st.write("This is a binary or non-numeric column. No kernel density plot available.")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Correlation Heatmap
st.write("### Correlation Heatmap")

numeric_data = data.select_dtypes(include=['number'])  # Select only numeric columns
corr_matrix = numeric_data.corr()

# Create an interactive heatmap using Plotly
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,  # Keep original correlation values
    x=[column_labels.get(col, col) for col in corr_matrix.columns],  # Use column_labels for x-axis labels
    y=[column_labels.get(col, col) for col in corr_matrix.columns],  # Use column_labels for y-axis labels
    colorscale="Viridis",  # Color scale
    colorbar=dict(title="Correlation", x=1.15),  # Add color scale to the right
    hoverongaps=False,  # To remove gaps when hovering
    hoverinfo="z+x+y",  # Display correlation values and row/column names on hover
    hoverlabel=dict(bgcolor="#282828", font=dict(color="white")),  # Box color followed by text color
    showscale=True,  # Show the color scale bar
))

# Create a slider for the user to select the correlation threshold
threshold = st.slider("Select Correlation Threshold", min_value=0.0, max_value=1.0, value=0.30, step=0.01)

# Update the z values to NaN where correlations are outside the desired range
fig.data[0].z[abs(corr_matrix) <= threshold] = None

# Update the layout of the heatmap to add gridlines
fig.update_layout(
    title="Correlation Heatmap",
    xaxis_nticks=len(corr_matrix.columns),
    yaxis_nticks=len(corr_matrix.columns),
    xaxis_title="Features",
    yaxis_title="Features",
    title_x=0.50,  # Center the title horizontally
    title_y=0.95,  # Position the title closer to the top
    title_xanchor="center",  # Center the title horizontally
    title_yanchor="top",  # Position the title at the top
    height=1000,  # Set the desired height (e.g., 400 pixels)
    width=1200,  # Set the desired width (e.g., 600 pixels)
    # xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),  # Add x-axis gridlines
    # yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),  # Add y-axis gridlines
)

# Show the interactive plot
st.plotly_chart(fig)


########################################################################################################################
# Data Privacy Information
st.write("### Data Privacy Information")
st.write("We are committed to maintaining data privacy. Customer financial data is not shared or exposed.")

####
####
# add in:

#Streamlit Caching: Use Streamlit's caching feature (@st.cache) to cache expensive computations or data loading operations. This way, these operations won't be re-executed unnecessarily on each interaction.

#Profiling: Use Python profiling tools to identify performance bottlenecks in your code. The cProfile module can help you understand which functions are taking the most time.

#Concurrency: If your app involves multiple computationally expensive tasks, consider using Streamlit's st.spinner and st.empty() to indicate when the app is processing and to avoid blocking the UI.

