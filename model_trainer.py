# model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
# Import functions for standalone testing
from data_loader import load_data
from data_preprocessor import clean_and_preprocess

def train_and_evaluate_models(df):
    """
    Defines, trains, and evaluates multiple classification models.
    Returns the results, feature importances, and test data for visualization.
    """
    if df is None:
        return None, None, None, None

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }

    results = {}
    feature_importances = {}

    print("\n--- Starting Model Training and Evaluation ---")
    for model_name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        print(f"Training {model_name}...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'pipeline': pipeline
        }
        
        print(f"\n--- Results for {model_name} ---")
        print(f"Accuracy: {results[model_name]['accuracy']:.4f}")
        print(f"ROC-AUC Score: {results[model_name]['roc_auc']:.4f}")
        print("Confusion Matrix:")
        print(results[model_name]['confusion_matrix'])
        print("Classification Report:")
        print(results[model_name]['classification_report'])

        if hasattr(model, 'feature_importances_'):
            ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
            all_feature_names = np.concatenate([numerical_cols, ohe_feature_names])
            importances = pd.Series(model.feature_importances_, index=all_feature_names)
            feature_importances[model_name] = importances.sort_values(ascending=False)

    print("\n--- Model Training and Evaluation Complete ---")
    return results, feature_importances, X_test, y_test

# This block allows the script to be run directly for testing.
if __name__ == '__main__':
    filepath = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # Load and preprocess the data
    raw_df = load_data(filepath)
    cleaned_df = clean_and_preprocess(raw_df)
    
    # Train and evaluate the models
    if cleaned_df is not None:
        # We don't need to store the results here, just run the function
        train_and_evaluate_models(cleaned_df)
