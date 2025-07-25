# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
# Import functions for standalone testing
from data_loader import load_data
from data_preprocessor import clean_and_preprocess

def run_eda(df):
    """
    Performs and displays Exploratory Data Analysis plots.
    """
    if df is None:
        return

    print("--- Starting Exploratory Data Analysis ---")
    sns.set_style("whitegrid")

    # Churn Distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Churn', data=df, palette='viridis')
    plt.title('Churn Distribution (0 = No, 1 = Yes)')
    plt.show()

    # Churn by Categorical Features
    categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 18))
    axes = axes.flatten()
    for i, col in enumerate(categorical_features):
        sns.countplot(x=col, hue='Churn', data=df, ax=axes[i], palette='pastel')
        axes[i].set_title(f'Churn by {col}')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xlabel('')
    plt.tight_layout()
    plt.show()

    # Churn by Numerical Features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    axes = axes.flatten()
    for i, col in enumerate(numerical_features):
        sns.histplot(data=df, x=col, hue='Churn', multiple='stack', ax=axes[i], palette='magma', kde=True)
        axes[i].set_title(f'Churn Distribution by {col}')
    plt.tight_layout()
    plt.show()
    print("--- EDA Complete ---")


def visualize_model_comparison(results, X_test, y_test):
    """
    Generates and displays ROC curves and confusion matrices for model comparison.
    """
    print("--- Generating Model Comparison Visualizations ---")
    # ROC Curves
    plt.figure(figsize=(10, 8))
    for model_name, result in results.items():
        y_pred_proba = result['pipeline'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {result['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (model_name, result) in enumerate(results.items()):
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {model_name}')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
    print("--- Model Comparison Visualizations Complete ---")


def visualize_feature_importance(feature_importances):
    """
    Generates and displays feature importance plots.
    """
    print("--- Generating Feature Importance Visualizations ---")
    for model_name, importances in feature_importances.items():
        plt.figure(figsize=(12, 8))
        importances.nlargest(20).plot(kind='barh', color='skyblue')
        plt.title(f'Top 20 Feature Importances for {model_name}')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.show()

# This block allows the script to be run directly for testing the EDA function.
if __name__ == '__main__':
    filepath = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # Load and preprocess the data
    raw_df = load_data(filepath)
    cleaned_df = clean_and_preprocess(raw_df)
    
    # Run the EDA visualization
    if cleaned_df is not None:
        run_eda(cleaned_df)
