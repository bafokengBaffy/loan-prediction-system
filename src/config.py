# config.py
import os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
CURRENT_DATE = datetime.now().strftime("%Y%m%d")

CONFIG = {
    "data": {
        "raw_path": os.path.join(BASE_DIR, "data/loan_data_set.csv"),
        "external_sources": [
            "https://api.creditbureau.com/scores",
            "data/external/economic_indicators.csv"
        ],
        "error_log_path": os.path.join(BASE_DIR, "data/error_logs/errors.csv"),
        "features": {
            "numeric": [
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                'Total_Income', 'EMI', 'Balance_Income', 'Loan_to_Income_Ratio',
                'LoanAmount_log', 'Income_per_Dependent', 'Credit_History_Income_Interaction'
            ],
            "categorical": [
                'Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Credit_History', 'Property_Area'
            ],
            "temporal": "Loan_Application_Date",
            "target": "Loan_Status",
            "id_column": "Loan_ID"
        },
        "augmentation": {
            "enable": True,
            "ratio": 0.3,
            "methods": ["smote", "adasyn"]
        }
    },
    "model": {
        "output_dir": os.path.join(BASE_DIR, "models"),
        "best_model_name": "best_model.pkl",
        "metadata_name": "model_metadata.json",
        "current_version": f"v4.2.1_{CURRENT_DATE}",
        "monitoring_metrics": {
            "primary": "balanced_accuracy",
            "secondary": ["roc_auc", "f1", "precision_at_90recall"],
            "business": ["approval_rate", "avg_loan_size"]
        },
        "params": {
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5,
            "stacking": {
                "final_estimator": "XGBClassifier",
                "level_1_estimators": [
                    ("rf", "RandomForestClassifier"),
                    ("svm", "SVC"),
                    ("gbm", "GradientBoostingClassifier")
                ]
            },
            "bayes_search": {
                "n_iter": 200,
                "scoring": "roc_auc_ovr_weighted",
                "param_space": {
                    'classifier__n_estimators': (100, 1000),
                    'classifier__max_depth': (3, 15),
                    'classifier__learning_rate': (0.001, 0.5, 'log-uniform'),
                    'classifier__subsample': (0.5, 1.0),
                    'classifier__colsample_bytree': (0.5, 1.0),
                    'classifier__gamma': (0, 2),
                    'classifier__reg_alpha': (0, 10),
                    'classifier__reg_lambda': (0, 10),
                    'classifier__scale_pos_weight': [1, 10, 25]
                }
            },
            "sampling": {
                "strategy": "SMOTETomek",
                "params": {
                    "smote": {"sampling_strategy": 0.5},
                    "tomek": {"sampling_strategy": 'majority'}
                }
            }
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "loan_approval",
            "registered_model_name": "LoanApprovalStackedEnsemble"
        }
    },
    "app": {
        "page_title": "Loan Approval Decision System",
        "page_icon": "💰",
        "layout": "wide",
        "sidebar_state": "expanded",
        "retrain_interval": 100,
        "content": {
            "header_title": "LOAN APPROVAL DECISION SYSTEM",
            "header_subtitle": "AI-powered decision support system",  # Added this line
            "system_status": {
                "model_version": f"v4.2.1_{CURRENT_DATE}",
                "decision_speed": "42ms avg",
                "compliance": ["FCRA", "ECOA", "Fair Lending"]
            },
            "contact_info": {
                "phone": "+1 (555) 123-4567",
                "email": "support@loandecisions.ai"
            },
            "footer": {
                "copyright": "© 202 Loan Decision Systems Inc.",
                "disclaimer": "This system provides recommendations only. Final decisions require human review.",
                "note": f"v4.2.1 | Last updated: {CURRENT_DATE}"  # Fixed f-string
            }
        },
        "compliance_rules": {
            "min_age": 18,
            "max_debt_to_income": 0.45
        },
        "monitoring": {
            "drift_threshold": 0.15,
            "performance_check_interval": 24  # hours
        }, 
        "database": {
        "host": "localhost",
        "name": "loan",
        "user": "root",
        "password": "1111",
        "port": 3306
    }
    }
}