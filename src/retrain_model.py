# retrain_model.py
import schedule
import time
from train_model import perform_training
from data_processing import process_new_submissions
from database import init_db

def retrain_job():
    print("Starting scheduled retraining...")
    conn = init_db()
    
    # First process any new submissions
    if process_new_submissions(conn):
        print("New submissions processed successfully")
    else:
        print("No new submissions to process")
    
    # Then retrain the model
    result = perform_training()
    if result["status"] == "success":
        print(f"Retraining successful! New accuracy: {result['metrics']['accuracy']:.1%}")
    else:
        print(f"Retraining failed: {result['error']}")
    
    conn.close()

if __name__ == "__main__":
    # Run daily at 2 AM
    schedule.every().day.at("02:00").do(retrain_job)
    
    while True:
        schedule.run_pending()
        time.sleep(60)