
import pandas as pd

# Load original data
original_data = pd.read_csv('../Data/Base.csv')

# Define the number of rows
subset_size = 10000

# Randomly select a subset of the data
testing_data = original_data.sample(n=subset_size, random_state=42)  # Set a random seed for reproducibility

# Save the testing dataset
testing_data.to_csv('subset_viz_data.csv', index=False)