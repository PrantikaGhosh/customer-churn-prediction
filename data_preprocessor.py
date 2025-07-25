# data_preprocessor.py
import pandas as pd
# We need to import the loader function to use it for testing
from data_loader import load_data 

def clean_and_preprocess(df):
    """
    Cleans and preprocesses the raw Telco Churn DataFrame.
    """
    if df is None:
        return None

    # The 'TotalCharges' column has missing values for new customers and is of object type.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    # The 'customerID' column is a unique identifier and not a predictive feature.
    # Use a try-except block in case the column was already dropped.
    try:
        df.drop('customerID', axis=1, inplace=True)
    except KeyError:
        pass # Column already dropped, do nothing.

    # The target variable 'Churn' is categorical ('Yes'/'No').
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    print("--- Data after Cleaning ---")
    print(df.head())
    print(f"\nTotal missing values after cleaning: {df.isnull().sum().sum()}")
    print("\nData cleaning and preprocessing complete.")
    return df

# This block allows the script to be run directly for testing.
if __name__ == '__main__':
    # Define the path to your dataset file
    filepath = "C:/Users/prant/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # Step 1: Load the data using the function from the other file
    raw_df = load_data(filepath)
    
    # Step 2: If data is loaded, pass it to the cleaning function
    if raw_df is not None:
        cleaned_df = clean_and_preprocess(raw_df)

