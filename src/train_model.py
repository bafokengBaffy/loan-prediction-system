#train_model.py
import os
import pandas as pd
import numpy as np
import mlflow
import joblib
import logging
import json
from typing import Dict, Any
from tempfile import mkdtemp
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn imports
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                           roc_auc_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as imbalanced_Pipeline

# Models and optimization
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from config import CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedLoanModelTrainer:
    """Enhanced model training pipeline with accuracy improvements"""
    
    def __init__(self):
        self.random_state = CONFIG["model"]["params"]["random_state"]
        np.random.seed(self.random_state)
        self.cache_dir = mkdtemp()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(CONFIG["model"]["mlflow"]["tracking_uri"])
        mlflow.set_experiment(CONFIG["model"]["mlflow"]["experiment_name"])
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate input data with enhanced checks"""
        try:
            data_path = CONFIG["data"]["raw_path"]
            logger.info(f"Loading data from {data_path}")
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
                
            data = pd.read_csv(data_path)
            
            # Enhanced column validation
            required_cols = {
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                'Loan_Amount_Term', 'Credit_History', 'Gender', 
                'Married', 'Dependents', 'Education', 'Self_Employed', 
                'Property_Area', 'Loan_Status'
            }
            missing_cols = required_cols - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            return data
            
        except Exception as e:
            logger.error("Data loading failed: %s", str(e))
            raise RuntimeError(f"Data loading failed: {str(e)}") from e
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced preprocessing with feature engineering"""
        try:
            logger.info("Starting data preprocessing")
            data = data.copy()
            
            # Clean column names
            data.columns = data.columns.str.replace(r'[\/, ]', '_', regex=True)
            
            # Handle missing values
            data['Dependents'] = data['Dependents'].replace('3+', '3').astype('float32')
            
            # Imputation
            num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
            cat_cols = ['Gender', 'Married', 'Self_Employed']
            
            # Handle zero values in Loan_Amount_Term
            data['Loan_Amount_Term'] = data['Loan_Amount_Term'].replace(0, np.nan)
            
            num_imputer = IterativeImputer(random_state=self.random_state,
                                         min_value=1,
                                         max_value=480)
            data[num_cols] = num_imputer.fit_transform(data[num_cols])
            
            data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
            
            # Feature engineering - ensure all features in config are created
            data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
            data['Loan_to_Income_Ratio'] = data['LoanAmount'] / (data['Total_Income'] + 1e-6)
            data['EMI'] = data['LoanAmount'] / data['Loan_Amount_Term']
            data['Balance_Income'] = data['Total_Income'] - (data['EMI'] * 1000)
            data['LoanAmount_log'] = np.log1p(data['LoanAmount'])
            data['Income_per_Dependent'] = data['Total_Income'] / (data['Dependents'].replace(0, 1))  # Handle zero dependents
            data['Credit_History_Income_Interaction'] = data['Credit_History'] * data['Total_Income']
            data['Loan_Term_to_Income_Ratio'] = data['Loan_Amount_Term'] / data['Total_Income']
            
            # Verify all features exist
            expected_features = CONFIG["data"]["features"]["numeric"] + CONFIG["data"]["features"]["categorical"]
            missing_features = set(expected_features) - set(data.columns)
            if missing_features:
                raise ValueError(f"Missing engineered features: {missing_features}")
            
            return data
            
        except Exception as e:
            logger.error("Data preprocessing failed: %s", str(e))
            raise RuntimeError(f"Data preprocessing failed: {str(e)}") from e
    
    def create_preprocessor(self) -> ColumnTransformer:
        """Create preprocessing pipeline"""
        numeric_features = CONFIG["data"]["features"]["numeric"]
        categorical_features = CONFIG["data"]["features"]["categorical"]
        
        numeric_transformer = Pipeline([
            ('imputer', IterativeImputer(random_state=self.random_state)),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        return ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    def create_model(self):
        """Create optimized XGBoost classifier"""
        return XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            tree_method='hist',
            n_jobs=-1,
            use_label_encoder=False
        )
    
    def create_sampler(self):
        """Create advanced sampler"""
        return SMOTETomek(
            sampling_strategy='auto',
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def evaluate_model(self, model, X_test, y_test):
        """Model evaluation with comprehensive metrics"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred))
        }
        
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), 
                    annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig("confusion_matrix.png")
        plt.close()
        
        return metrics, clf_report
    
    def train_model(self) -> Dict[str, Any]:
        """Main training pipeline"""
        with mlflow.start_run():
            try:
                # Load and process data
                data = self.load_data()
                processed_data = self.preprocess_data(data)
                
                X = processed_data.drop(['Loan_Status', 'Loan_ID'], axis=1, errors='ignore')
                y = processed_data['Loan_Status'].map({'Y': 1, 'N': 0})
                
                # Verify features before train-test split
                expected_features = CONFIG["data"]["features"]["numeric"] + CONFIG["data"]["features"]["categorical"]
                missing_features = set(expected_features) - set(X.columns)
                if missing_features:
                    raise ValueError(f"Missing features in X: {missing_features}")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=CONFIG["model"]["params"]["test_size"],
                    random_state=self.random_state,
                    stratify=y
                )
                
                pipeline = imbalanced_Pipeline([
                    ('preprocessor', self.create_preprocessor()),
                    ('sampler', self.create_sampler()),
                    ('classifier', self.create_model())
                ])
                
                param_space = {
                    'classifier__n_estimators': Integer(100, 1000),
                    'classifier__max_depth': Integer(3, 10),
                    'classifier__learning_rate': Real(0.001, 0.3, 'log-uniform'),
                    'classifier__subsample': Real(0.6, 1.0),
                    'classifier__colsample_bytree': Real(0.6, 1.0),
                    'classifier__gamma': Real(0, 5),
                    'classifier__reg_alpha': Real(0, 10),
                    'classifier__reg_lambda': Real(0, 10)
                }
                
                opt = BayesSearchCV(
                    pipeline,
                    param_space,
                    n_iter=50,
                    cv=StratifiedKFold(
                        CONFIG["model"]["params"]["cv_folds"],
                        shuffle=True,
                        random_state=self.random_state
                    ),
                    n_jobs=-1,
                    scoring='roc_auc',
                    random_state=self.random_state,
                    verbose=2
                )
                
                logger.info("Starting model training")
                opt.fit(X_train, y_train)
                
                # Get processed feature names
                preprocessor = opt.best_estimator_.named_steps['preprocessor']
                numeric_features = CONFIG["data"]["features"]["numeric"]
                
                # Handle categorical features
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
                if hasattr(cat_encoder, 'get_feature_names_out'):
                    categorical_features = cat_encoder.get_feature_names_out(
                        CONFIG["data"]["features"]["categorical"]
                    )
                else:
                    categorical_features = []
                    for i, col in enumerate(CONFIG["data"]["features"]["categorical"]):
                        for cat in cat_encoder.categories_[i]:
                            categorical_features.append(f"{col}_{cat}")
                
                all_features = list(numeric_features) + list(categorical_features)
                feature_importances = opt.best_estimator_.named_steps['classifier'].feature_importances_
                
                # Validate feature importance dimensions
                if len(all_features) != len(feature_importances):
                    logger.warning("Feature dimension mismatch, using numeric features only")
                    all_features = numeric_features
                    feature_importances = feature_importances[:len(numeric_features)]
                
                importance_df = pd.DataFrame({
                    'Feature': all_features,
                    'Importance': feature_importances
                }).sort_values('Importance', ascending=False)
                
                # Plot and save feature importance
                plt.figure(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
                plt.title('Top 20 Feature Importances')
                plt.tight_layout()
                plt.savefig("feature_importance.png")
                plt.close()
                
                # Evaluate and log results
                metrics, clf_report = self.evaluate_model(opt.best_estimator_, X_test, y_test)
                
                mlflow.log_params({str(k): str(v) for k, v in opt.best_params_.items()})
                mlflow.log_metrics(metrics)
                mlflow.log_artifact("confusion_matrix.png")
                mlflow.log_artifact("feature_importance.png")
                mlflow.log_dict(clf_report, "classification_report.json")
                
                # Save model artifacts
                model_dir = CONFIG["model"]["output_dir"]
                os.makedirs(model_dir, exist_ok=True)
                
                model_path = os.path.join(model_dir, CONFIG["model"]["best_model_name"])
                joblib.dump(opt.best_estimator_, model_path)
                
                metadata = {
                    "metrics": metrics,
                    "best_params": str(opt.best_params_),
                    "feature_importances": importance_df.to_dict(),
                    "classification_report": clf_report
                }
                
                with open(os.path.join(model_dir, CONFIG["model"]["metadata_name"]), 'w') as f:
                    json.dump(metadata, f)
                
                logger.info(f"Model saved to: {model_path}")
                return {"status": "success", "model_path": model_path, "metrics": metrics}
                
            except Exception as e:
                logger.error("Training failed: %s", str(e), exc_info=True)
                mlflow.log_param("error", str(e))
                return {"status": "error", "error": str(e)}

def perform_training() -> Dict[str, Any]:
    """Public interface for model training"""
    try:
        trainer = EnhancedLoanModelTrainer()
        result = trainer.train_model()
        return result
    except Exception as e:
        logger.error("Training process failed: %s", str(e))
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    result = perform_training()
    if result["status"] == "success":
        print(f"✅ Training successful! Metrics: {result['metrics']}")
    else:
        print(f"❌ Training failed: {result['error']}")