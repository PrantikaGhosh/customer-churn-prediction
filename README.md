# Customer Churn Prediction using Machine Learning

## Project Overview

This project demonstrates a complete machine learning workflow for predicting customer churn using the "Telco Customer Churn" dataset from Kaggle. It involves data loading, cleaning, exploratory data analysis (EDA), model training, and evaluation. The project compares four different classification models to determine the most effective one for this dataset:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

The code is structured into modular Python files for better organization and reusability.

## Features

- **Modular Code:** The project is broken down into separate files for data loading, preprocessing, visualization, and model training.
- **Data Cleaning:** Handles missing values and converts data into a usable format.
- **Exploratory Data Analysis (EDA):** Generates visualizations to understand the relationships between different features and customer churn.
- **Model Comparison:** Trains and evaluates four different classification algorithms.
- **Performance Evaluation:** Uses key metrics like Accuracy, ROC-AUC Score, Confusion Matrix, and a detailed Classification Report to assess model performance.
- **Feature Importance:** Visualizes the most influential factors for tree-based models (Random Forest, XGBoost) to understand what drives churn.

## Project Structure

```
.
├── venv/
├── data_loader.py          # Loads the dataset
├── data_preprocessor.py    # Cleans and preprocesses the data
├── model_trainer.py        # Trains and evaluates the models
├── visualizer.py           # Handles all visualizations (EDA, results)
├── main.py                 # Main script to run the entire pipeline
├── requirements.txt        # Lists all project dependencies
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # The dataset file
└── README.md               # This file
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- A virtual environment tool (like `venv`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PrantikaGhosh/customer-churn-prediction.git](https://github.com/PrantikaGhosh/customer-churn-prediction.git)
    cd customer-churn-prediction
    ```

2.  **Create and activate a virtual environment:**
    * On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\Activate.ps1
        ```
    * On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    Download the dataset from [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file in the root directory of the project.

### Usage

To run the entire pipeline (from data loading to model evaluation and visualization), execute the `main.py` script:

```bash
python main.py
```

The script will print the results of the model evaluations to the console and display several plots for the EDA and model comparisons.

## Model Performance

Based on the evaluation, the **Logistic Regression** model performed the best on this dataset, achieving the highest accuracy and ROC-AUC score.

| Model               | Accuracy | ROC-AUC Score |
| ------------------- | -------- | ------------- |
| **Logistic Regression** | **80.55%** | **0.8421** |
| Random Forest       | 78.64%   | 0.8185      |
| SVM                 | 79.13%   | 0.7902      |
| XGBoost             | 77.29%   | 0.8152      |
