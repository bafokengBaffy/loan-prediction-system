# database.py
import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime
from config import CONFIG

def init_db():
    """Initialize the database"""
    db_path = Path(CONFIG["data"]["raw_path"]).parent / "user_submissions.db"
    conn = sqlite3.connect(db_path)
    
    # Create table if it doesn't exist
    conn.execute("""
    CREATE TABLE IF NOT EXISTS submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        gender TEXT,
        married TEXT,
        dependents TEXT,
        education TEXT,
        self_employed TEXT,
        applicant_income REAL,
        coapplicant_income REAL,
        loan_amount REAL,
        loan_amount_term REAL,
        credit_history TEXT,
        property_area TEXT,
        prediction INTEGER,
        probability REAL,
        final_decision TEXT,
        user_feedback TEXT,
        processed BOOLEAN DEFAULT 0
    )
    """)
    conn.commit()
    return conn

def save_submission(conn, input_data, prediction, probability):
    """Save a new submission to the database"""
    try:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO submissions (
            timestamp, gender, married, dependents, education, 
            self_employed, applicant_income, coapplicant_income, 
            loan_amount, loan_amount_term, credit_history, property_area,
            prediction, probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            input_data.get('Gender'),
            input_data.get('Married'),
            input_data.get('Dependents'),
            input_data.get('Education'),
            input_data.get('Self_Employed'),
            input_data.get('ApplicantIncome'),
            input_data.get('CoapplicantIncome'),
            input_data.get('LoanAmount'),
            input_data.get('Loan_Amount_Term'),
            input_data.get('Credit_History'),
            input_data.get('Property_Area'),
            prediction,
            probability
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving submission: {e}")
        return False

def get_unprocessed_submissions(conn):
    """Retrieve submissions that haven't been added to training data"""
    df = pd.read_sql("""
    SELECT * FROM submissions 
    WHERE processed = 0
    """, conn)
    return df

def mark_as_processed(conn, submission_ids):
    """Mark submissions as processed"""
    if not submission_ids:
        return
    placeholders = ','.join(['?'] * len(submission_ids))
    conn.execute(f"""
    UPDATE submissions 
    SET processed = 1 
    WHERE id IN ({placeholders})
    """, submission_ids)
    conn.commit()