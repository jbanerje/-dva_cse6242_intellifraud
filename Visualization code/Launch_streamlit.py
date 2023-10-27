import subprocess

# Streamlit app file
app_file = 'EDA_Streamit_v1.py'

# Define the command to run Streamlit
streamlit_command = f'streamlit run {app_file}'

# Launch Streamlit in the browser
subprocess.Popen(streamlit_command, shell=True)
