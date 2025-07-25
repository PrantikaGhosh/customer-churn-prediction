# data_loader.py
import pandas as pd

def load_data(filepath):
    """
    Loads the Telco Customer Churn dataset from a specified filepath.
    """
    try:
        df = pd.read_csv("C:/Users/prant/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        print("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        print("Please download the dataset from Kaggle and place it in the correct directory.")
        return None

# This block allows the script to be run directly for testing.
if __name__ == '__main__':
    # Define the path to your dataset file
    filepath = "C:/Users/prant/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # Call the function to load the data
    churn_df = load_data(filepath)
    
    # If the data is loaded successfully, print the first few rows.
    if churn_df is not None:
        print("\n--- Data Head ---")
        print(churn_df.head())
