# compliance.py
from datetime import datetime, timedelta
from database import init_db

def anonymize_old_submissions():
    """Anonymize submissions older than retention period"""
    conn = init_db()
    cutoff_date = (datetime.now() - timedelta(days=365)).isoformat()
    
    conn.execute("""
    UPDATE submissions
    SET 
        gender = 'Anonymous',
        applicant_income = NULL,
        coapplicant_income = NULL,
        loan_amount = NULL,
        property_area = NULL
    WHERE timestamp < ? AND processed = 1
    """, (cutoff_date,))
    
    conn.commit()
    conn.close()

def delete_sensitive_data(submission_ids):
    """Permanently delete sensitive data for specific submissions"""
    conn = init_db()
    placeholders = ','.join(['?'] * len(submission_ids))
    
    conn.execute(f"""
    DELETE FROM submissions
    WHERE id IN ({placeholders})
    """, submission_ids)
    
    conn.commit()
    conn.close()