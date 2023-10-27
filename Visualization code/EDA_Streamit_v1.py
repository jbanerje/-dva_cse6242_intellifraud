import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go



# with st.expander("See explanation"):
#     st.write("test")
#     -- This will create a button that allows the user to open an expandable section below the figure. Use
#     to add a description or anything where you wish to display text under a figure

# Load dataset
data = pd.read_csv('subset_viz_data.csv')  # Replace with your dataset file

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

st.set_option('deprecation.showPyplotGlobalUse', False)

# Add Streamlit code to create the interactive dashboard
st.title('Exploratory Data Analysis')

# Add widgets (e.g., sliders, dropdowns, etc.) for user interaction
selected_feature = st.selectbox('Select a feature for analysis', list(column_labels.values()))


# Display basic statistics
selected_column = [col for col, label in column_labels.items() if label == selected_feature][0]
st.write(f"### Descriptive Statistics for {selected_feature} ({selected_column})")

# Check if the selected column is numeric before calculating statistics
if data[selected_column].dtype in ['int64', 'float64']:
    st.write(data[selected_column].describe())
else:
    st.write("This is a non-numeric column. No statistics available.")

# Create a violin plot for the selected feature using Plotly
st.write(f"### {selected_feature} Distribution ({selected_column})")
fig = go.Figure()

# Check if the selected column is numeric before creating the violin plot
if data[selected_column].dtype in ['int64', 'float64']:
    fig.add_trace(go.Violin(y=data[selected_column], box_visible=True, line_color="purple"))
    custom_tooltip = f"Mean: {data[selected_column].mean():.2f}<br>Median: {data[selected_column].median():.2f}"
    fig.update_traces(hoverinfo='y+name', name=custom_tooltip, line_color="purple")
else:
    st.write("This is a non-numeric column. No violin plot available.")

st.plotly_chart(fig)

# Check if the selected column is numeric before creating the histogram
if data[selected_column].dtype in ['int64', 'float64']:
    hist_fig = px.histogram(data, x=selected_column, nbins=30)
    hist_fig.update_layout(
        title=f"{selected_feature} Distribution",
        xaxis_title=selected_column,
        yaxis_title="Frequency",
        hovermode="closest",  # Enable hover for data points
        hoverlabel=dict(bgcolor="white", bordercolor="gray"),
    )
    hist_fig.update_traces(
        hoverinfo="x+y+name",  # Display x, y, and name (custom tooltip) on hover
        name=custom_tooltip,
        hoverlabel = dict(bgcolor="#282828", font = dict(color="white")) # Box color followed by text color
    )
    # Display the histogram
    st.plotly_chart(hist_fig)
else:
    st.write("This is a non-numeric column. No histogram available.")

# Create histograms
# st.write(f"### {selected_feature} Distribution ({selected_column})")
# fig, ax = plt.subplots()
# sns.histplot(data[selected_column], bins=30, kde=True, ax=ax)
# st.pyplot(fig)


#Correlation Heatmap
# st.write("### Correlation Heatmap")
# numeric_data = data.select_dtypes(include=['number'])  # Select only numeric columns
# corr_matrix = numeric_data.corr()
# fig, ax = plt.subplots(figsize=(20,20))
#
# # Adjust the font size for the axis labels and legend
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax,
#             xticklabels=corr_matrix.columns.map(column_labels),
#             yticklabels=corr_matrix.columns.map(column_labels),
#             cbar_kws={"shrink": 0.7})  # You can adjust the "shrink" parameter to resize the legend
#
# st.pyplot(fig)

########################################################################################################################
# Correlation Heatmap

st.write("### Correlation Heatmap")

numeric_data = data.select_dtypes(include=['number'])  # Select only numeric columns
corr_matrix = numeric_data.corr()

# Create an interactive heatmap using Plotly
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale="Viridis",  # Color scale
    colorbar=dict(title="Correlation"),
    hoverongaps=False,  # To remove gaps when hovering
    hoverinfo="z+x+y",  # Display correlation values and row/column names on hover
    showscale=False  # To hide the color scale bar
))

# Update the layout of the heatmap
fig.update_layout(
    title="Correlation Heatmap",
    xaxis_nticks=len(corr_matrix.columns),
    yaxis_nticks=len(corr_matrix.columns),
    xaxis_title="Features",
    yaxis_title="Features",
    height=1000,  # Set the desired height (e.g., 400 pixels)
    width=1200,  # Set the desired width (e.g., 600 pixels)
)

# Display the interactive heatmap
st.plotly_chart(fig)

########################################################################################################################
# Interactive Scatter Plot
st.write("### Interactive Scatter Plot")
x_feature = st.selectbox('Select X-axis feature', list(column_labels.values()))
y_feature = st.selectbox('Select Y-axis feature', list(column_labels.values()))
x_column = [col for col, label in column_labels.items() if label == x_feature][0]
y_column = [col for col, label in column_labels.items() if label == y_feature][0]
fig, ax = plt.subplots()

st.pyplot(fig)
########################################################################################################################

# Interactive Pair Plot
st.write("### Interactive Pair Plot")
numeric_columns = data.select_dtypes(include=['number']).columns.tolist()  # Get a list of numeric columns
pairplot_features = st.multiselect('Select features for Pair Plot', list(column_labels.values()))

if pairplot_features:
    pairplot_columns = [col for col, label in column_labels.items() if label in pairplot_features]
    pairplot_data = data[pairplot_columns]
    pairplot = sns.pairplot(data=pairplot_data, diag_kind="kde")
    st.pyplot(pairplot)
else:
    st.write("Select at least one numeric feature for the Pair Plot.")

########################################################################################################################
# Fraud Detection Model Insights (you can replace this with your model's output)
# st.write("### Fraud Detection Model Insights")
# Add information about your model's performance, confusion matrix, etc.

# Custom Insights (allow users to input their custom code for analysis)
# st.write("### Custom Insights")
# custom_code = st.text_area("Enter custom Python code for analysis")
# if st.button("Run Custom Code"):
#     try:
#         exec(custom_code)
#     except Exception as e:
#         st.error(f"An error occurred: {e}")

# Data Export Option
if st.button("Export Data"):
    st.write("Add code to export data to a file here")

# Data Privacy Information
st.write("### Data Privacy Information")
st.write("We are committed to maintaining data privacy. Customer financial data is not shared or exposed.")



