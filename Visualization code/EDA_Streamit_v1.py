import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

# # # MUST HAVE TO FORCE WIDE MODE! DO NOT MOVE!!! # # #
# Set the page layout
st.set_page_config(
    layout="wide",  # Use the wide layout to take full advantage of the screen width
    initial_sidebar_state="auto",  # Auto-hide the sidebar
)

# Define a fixed size for the plots
plot_width = 700
plot_height = 700

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
col3, col4 = st.columns(2)

# # # # # # # # # # # # Uncomment to get back basic descriptive stats table # # # # # # # # # # # #
# st.write(f"### Descriptive Statistics for {selected_feature}")

# Check if the selected column is numeric before calculating statistics
# if data[selected_column].dtype in ['int64', 'float64']:
#     st.write(data[selected_column].describe())
# else:
#     st.write("This is a non-numeric column. No statistics available.")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # Uncomment to get back basic violin plot # # # # # # # # # # # #

# Create a violin plot for the selected feature using Plotly
# st.write(f"### {selected_feature} Distribution ({selected_column})")
# fig = go.Figure()
#
# # Check if the selected column is numeric before creating the violin plot
# if data[selected_column].dtype in ['int64', 'float64']:
#     fig.add_trace(go.Violin(y=data[selected_column], box_visible=True, line_color="purple"))
#     custom_tooltip = f"Mean: {data[selected_column].mean():.2f}<br>Median: {data[selected_column].median():.2f}"
#     fig.update_traces(hoverinfo='y+name', name=custom_tooltip, line_color="purple")
# else:
#     st.write("This is a non-numeric column. No violin plot available.")
#
# st.plotly_chart(fig)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Color customization widgets for violin plot
with col1:
    #violin_color = st.color_picker("Select Violin Plot Color", value="#FFFFFF")
    #box_color = st.color_picker("Select Box Color", value="#000000")

    # Define default colors
    default_violin_color = "#FFFFFF"
    default_box_color = "#000000"

    # Create a violin plot for the selected feature using Plotly
  #  st.write(f"### {selected_feature} Distribution")

    # Check if the selected column is numeric before creating the violin plot
    if data[selected_column].dtype in ['int64', 'float64']:
        fig = go.Figure()

        fig.add_trace(go.Violin(
            y=data[selected_column],
            box_visible=True,
            line_color=default_box_color,
            fillcolor=default_violin_color,
            opacity=0.6
        ))

        custom_tooltip = f"Mean: {data[selected_column].mean():.2f}<br>Median: {data[selected_column].median():.2f}"
        fig.update_traces(hoverinfo='y+name', name=custom_tooltip, line_color=default_box_color)

        # Update layout and labels
        fig.update_layout(
            title=f"{selected_feature} Distribution",
            title_x=0.55,  # Center the title horizontally
            title_y=0.9,  # Position the title closer to the top
            title_xanchor="center",  # Center the title horizontally
            title_yanchor="top",  # Position the title at the top
            xaxis_title=selected_feature,
            yaxis_title="Density",
            width=plot_width,  # Set the width
            height=plot_height,  # Set the height
        )

        # Show the interactive plot
        st.plotly_chart(fig)

        # Move color selection boxes to the bottom
        # st.sidebar.header("Color Selection")
        # violin_color = st.sidebar.color_picker("Select Violin Plot Color", value="#FFFFFF")
        # box_color = st.sidebar.color_picker("Select Box Color", value="#000000")

    else:
        st.write("This is a non-numeric column. No violin plot available.")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Check if the selected column is numeric before creating the histogram
with col2:
    if data[selected_column].dtype in ['int64', 'float64']:
        hist_fig = px.histogram(data, x=selected_column, nbins=30, color_discrete_sequence=['lightblue'])  # Change the color here
        hist_fig.update_layout(
            title=f"{selected_feature} Histogram",
            title_x=0.55,  # Center the title horizontally
            title_y=0.92,  # Position the title closer to the top
            title_xanchor="center",  # Center the title horizontally
            title_yanchor="top",  # Position the title at the top
            xaxis_title=selected_feature,
            yaxis_title="Frequency",
            hovermode="closest",  # Enable hover for data points
            width=plot_width,  # Set the width
            height=plot_height,  # Set the height
            hoverlabel=dict(bgcolor="white", bordercolor="gray"),
        )
        hist_fig.update_traces(
            hoverinfo="x+y+name",  # Display x, y, and name (custom tooltip) on hover
            name=custom_tooltip,
            hoverlabel = dict(bgcolor="#282828", font = dict(color="white")) # Box color followed by text color
        )

        # Add padding to move the figure down
        st.markdown(
            f'<style>div.row-widget.stPlotlyChart {{ padding: 20px; }}</style>',
            unsafe_allow_html=True
        )

        # Display the histogram
        st.plotly_chart(hist_fig)
    else:
        st.write("This is a non-numeric column. No histogram available.")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Create a smooth Kernel Density Plot
# st.write(f"### {selected_feature} Kernel Density Plot")
# if data[selected_column].dtype in ['int64', 'float64']:
#     # Use seaborn to create the KDE plot
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(data[data['fraud_bool'] == 0][selected_column], label='Not Fraud', color='blue', shade=True)
#     sns.kdeplot(data[data['fraud_bool'] == 1][selected_column], label='Fraud', color='red', shade=True)
#     plt.xlabel(selected_column)
#     plt.ylabel("Density")
#     plt.title(f"Kernel Density Plot of {selected_feature}")
#     st.pyplot(plt)
# else:
#     st.write("This is a non-numeric column. No kernel density plot available.")


# Create a smooth Kernel Density Plot
# st.write(f"### {selected_feature} Kernel Density Plot")

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
        width=plot_width,  # Adjust the width of the figure
        height=plot_height,  # Adjust the height of the figure
    )


    # Show the interactive plot
    st.plotly_chart(fig)
else:
    st.write("This is a binary or non-numeric column. No kernel density plot available.")





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
# # Interactive Scatter Plot
# st.write("### Interactive Scatter Plot")
# x_feature = st.selectbox('Select X-axis feature', list(column_labels.values()))
# y_feature = st.selectbox('Select Y-axis feature', list(column_labels.values()))
# x_column = [col for col, label in column_labels.items() if label == x_feature][0]
# y_column = [col for col, label in column_labels.items() if label == y_feature][0]
# fig, ax = plt.subplots()
#
# st.pyplot(fig)
########################################################################################################################

# # Interactive Pair Plot
# st.write("### Interactive Pair Plot")
# numeric_columns = data.select_dtypes(include=['number']).columns.tolist()  # Get a list of numeric columns
# pairplot_features = st.multiselect('Select features for Pair Plot', list(column_labels.values()))
#
# if pairplot_features:
#     pairplot_columns = [col for col, label in column_labels.items() if label in pairplot_features]
#     pairplot_data = data[pairplot_columns]
#     pairplot = sns.pairplot(data=pairplot_data, diag_kind="kde")
#     st.pyplot(pairplot)
# else:
#     st.write("Select at least one numeric feature for the Pair Plot.")

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


########################################################################################################################

# Data Export Option
# if st.button("Export Data"):
#     st.write("Add code to export data to a file here")

# # # # Sample code to simulate data export # # #
# def export_data_to_csv(dataframe):
#     # Replace this with your export logic
#     filename = "exported_data.csv"
#     dataframe.to_csv(filename, index=False)
#     return filename
#
# st.title("Data Export Example")
#
# data = pd.DataFrame({'column1': [1, 2, 3, 4, 5], 'column2': [6, 7, 8, 9, 10]})
#
# # Data Export Option
# if st.button("Export Data"):
#     # Check the size of your data (use your actual data size)
#     data_size = data.memory_usage(deep=True).sum()  # Size in bytes
#
#     # Set a threshold size for prompting the user
#     threshold_size_bytes = 1024 * 1024  # 1 MB
#
#     if data_size > threshold_size_bytes:
#         # The data is larger than the threshold
#         confirmation = st.confirm(
#             "The exported data is large. Are you sure you want to export it?",
#             "Yes",
#             "No"
#         )
#         if confirmation:
#             st.success("Exporting data...")
#
#             # Call your export function
#             filename = export_data_to_csv(data)
#
#             # Provide a link to download the exported data
#             st.markdown(f"Download the exported data: [Exported Data](data:text/csv;base64,{filename})")
#         else:
#             st.warning("Data export canceled.")
#     else:
#         st.success("Exporting data...")
#
#         # Call your export function
#         filename = export_data_to_csv(data)
#
#         # Provide a link to download the exported data
#         st.markdown(f"Download the exported data: [Exported Data](data:text/csv;base64,{filename})")
########################################################################################################################



# Data Privacy Information
st.write("### Data Privacy Information")
st.write("We are committed to maintaining data privacy. Customer financial data is not shared or exposed.")



