# data_processing.py
import pandas as pd
from pathlib import Path
from database import get_unprocessed_submissions, mark_as_processed
from config import CONFIG

def process_new_submissions(conn):
    """Process new submissions and add to training data"""
    try:
        # Get unprocessed submissions
        new_data = get_unprocessed_submissions(conn)
        if new_data.empty:
            return False
        
        # Load existing data
        data_path = Path(CONFIG["data"]["raw_path"])
        existing_data = pd.read_csv(data_path)
        
        # Transform new data to match existing format
        new_data['Loan_Status'] = new_data['prediction'].map({1: 'Y', 0: 'N'})
        new_data['Loan_ID'] = 'USER_' + new_data['id'].astype(str)
        
        # Select and rename columns to match
        column_map = {
            'gender': 'Gender',
            'married': 'Married',
            'dependents': 'Dependents',
            'education': 'Education',
            'self_employed': 'Self_Employed',
            'applicant_income': 'ApplicantIncome',
            'coapplicant_income': 'CoapplicantIncome',
            'loan_amount': 'LoanAmount',
            'loan_amount_term': 'Loan_Amount_Term',
            'credit_history': 'Credit_History',
            'property_area': 'Property_Area',
            'Loan_Status': 'Loan_Status',
            'Loan_ID': 'Loan_ID'
        }
        
        processed_new = new_data[column_map.keys()].rename(columns=column_map)
        
        # Combine with existing data
        combined_data = pd.concat([existing_data, processed_new], ignore_index=True)
        
        # Save back to CSV
        combined_data.to_csv(data_path, index=False)
        
        # Mark submissions as processed
        mark_as_processed(conn, new_data['id'].tolist())
        
        return True
    except Exception as e:
        print(f"Error processing new submissions: {e}")
        return False