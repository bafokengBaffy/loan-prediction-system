import sys
sys.path.append("c:/users/baffk/appdata/roaming/python/python313/site-packages")
import pandas as pd # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
import joblib # type: ignore
import os

def train_and_save_model():
    # Load dataset
    try:
        data = pd.read_csv("data/loan_data_set.csv")
    except FileNotFoundError:
        print("Error: File not found. Please ensure:")
        print("1. The file 'loan_data_set.csv' exists in the 'data' folder")
        print("2. The file path is correct")
        exit()

    # Data Cleaning
    data = data.drop('Loan_ID', axis=1)
    data.columns = data.columns.str.replace(r'[\/, ]', '_', regex=True)

    # Convert 3+ to 3 in Dependents
    data['Dependents'] = data['Dependents'].replace('3+', '3')

    # Define features and target
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 
                          'Self_Employed', 'Credit_History', 'Property_Area']

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Split data
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status'].map({'Y': 1, 'N': 0})

    # Create and train model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ))
    ])

    model.fit(X, y)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/loan_model.pkl")
    print("Model trained and saved successfully")

if __name__ == "__main__":
    train_and_save_model()