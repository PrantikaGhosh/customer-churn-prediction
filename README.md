Customer Churn Prediction using Machine LearningProject OverviewThis project demonstrates a complete machine learning workflow for predicting customer churn using the "Telco Customer Churn" dataset from Kaggle. It involves data loading, cleaning, exploratory data analysis (EDA), model training, and evaluation. The project compares four different classification models to determine the most effective one for this dataset:Logistic RegressionRandom ForestSupport Vector Machine (SVM)XGBoostThe code is structured into modular Python files for better organization and reusability.FeaturesModular Code: The project is broken down into separate files for data loading, preprocessing, visualization, and model training.Data Cleaning: Handles missing values and converts data into a usable format.Exploratory Data Analysis (EDA): Generates visualizations to understand the relationships between different features and customer churn.Model Comparison: Trains and evaluates four different classification algorithms.Performance Evaluation: Uses key metrics like Accuracy, ROC-AUC Score, Confusion Matrix, and a detailed Classification Report to assess model performance.Feature Importance: Visualizes the most influential factors for tree-based models (Random Forest, XGBoost) to understand what drives churn.Project Structure.
├── venv/
├── data_loader.py          # Loads the dataset
├── data_preprocessor.py    # Cleans and preprocesses the data
├── model_trainer.py        # Trains and evaluates the models
├── visualizer.py           # Handles all visualizations (EDA, results)
├── main.py                 # Main script to run the entire pipeline
├── requirements.txt        # Lists all project dependencies
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # The dataset file
└── README.md               # This file
Getting StartedPrerequisitesPython 3.8 or higherA virtual environment tool (like venv)InstallationClone the repository:git clone <your-repository-url>
cd <your-repository-name>
Create and activate a virtual environment:On Windows:python -m venv venv
.\venv\Scripts\Activate.ps1
On macOS/Linux:python3 -m venv venv
source venv/bin/activate
Install the required dependencies:pip install -r requirements.txt
Download the dataset:Download the dataset from Kaggle: Telco Customer Churn and place the WA_Fn-UseC_-Telco-Customer-Churn.csv file in the root directory of the project.UsageTo run the entire pipeline (from data loading to model evaluation and visualization), execute the main.py script:python main.py
The script will print the results of the model evaluations to the console and display several plots for the EDA and model comparisons.Model PerformanceBased on the evaluation, the Logistic Regression model performed the best on this dataset, achieving the highest accuracy and ROC-AUC score.ModelAccuracyROC-AUC ScoreLogistic Regression80.55%0.8421Random Forest78.64%0.8185SVM79.13%0.7902XGBoost77.29%0.8152