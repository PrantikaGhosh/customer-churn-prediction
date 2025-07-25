# main.py
# This is the main script to run the churn prediction project.

# Import functions from other files
from data_loader import load_data
from data_preprocessor import clean_and_preprocess
from visualizer import run_eda, visualize_model_comparison, visualize_feature_importance
from model_trainer import train_and_evaluate_models

def main():
    """
    Main function to execute the entire churn prediction pipeline.
    """
    # Define the path to your dataset
    filepath = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

    # Step 1: Load Data
    raw_df = load_data(filepath)

    # Step 2: Clean and Preprocess Data
    cleaned_df = clean_and_preprocess(raw_df)
    
    # Step 3: Perform Exploratory Data Analysis
    # You can comment this out if you don't need to see the plots every time.
    run_eda(cleaned_df)

    # Step 4: Train models and get results
    results, feature_importances, X_test, y_test = train_and_evaluate_models(cleaned_df)

    # Step 5: Visualize the results
    if results:
        visualize_model_comparison(results, X_test, y_test)
        visualize_feature_importance(feature_importances)

    print("\n--- Churn Prediction Script Finished ---")

# This ensures the main function runs only when the script is executed directly
if __name__ == '__main__':
    main()
