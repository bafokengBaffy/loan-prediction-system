import mysql.connector
from mysql.connector import Error
import numpy as np
import pandas as pd
from datetime import datetime
from config import CONFIG
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MySQLDatabase:
    def __init__(self):
        self.config = CONFIG.get("database", {})
        self.host = self.config.get("host", "localhost")
        self.database = self.config.get("name", "loan")
        self.user = self.config.get("user", "root")
        self.password = self.config.get("password", "1111")
        self.port = self.config.get("port", 3306)
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                auth_plugin='mysql_native_password'
            )
            if self.connection.is_connected():
                logger.info(f"Connected to MySQL database: {self.database}")
                return self.connection
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise RuntimeError(f"""
            Database connection failed. Please check:
            1. MySQL server is running
            2. User '{self.user}' has proper permissions
            3. Password is correct
            4. Host '{self.host}' is accessible
            5. Database '{self.database}' exists

            Original error: {e}
            """) from e

    def init_db(self):
        """Only ensures the loan table exists; other tables should be managed manually."""
        conn = None
        cursor = None
        try:
            conn = self.connect()
            cursor = conn.cursor()

            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            cursor.execute(f"USE {self.database}")

            # Only creating the 'loan' table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS loan (
                id INT AUTO_INCREMENT PRIMARY KEY,
                loan_id VARCHAR(50) UNIQUE,
                timestamp DATETIME NOT NULL,
                gender VARCHAR(10),
                married VARCHAR(3),
                dependents VARCHAR(3),
                education VARCHAR(15),
                self_employed VARCHAR(3),
                applicant_income DECIMAL(12,2),
                coapplicant_income DECIMAL(12,2),
                loan_amount DECIMAL(12,2),
                loan_amount_term INT,
                credit_history VARCHAR(10),
                property_area VARCHAR(10),
                total_income DECIMAL(12,2),
                loan_to_income_ratio DECIMAL(10,4),
                emi DECIMAL(12,2),
                balance_income DECIMAL(12,2),
                loan_amount_log DECIMAL(10,4),
                income_per_dependent DECIMAL(12,2),
                credit_history_income_interaction DECIMAL(12,2),
                prediction TINYINT COMMENT '0=Rejected, 1=Approved',
                probability DECIMAL(5,4),
                actual_status VARCHAR(1),
                feedback TEXT,
                processed BOOLEAN DEFAULT FALSE,
                model_version VARCHAR(50),
                data_source VARCHAR(20),
                INDEX idx_processed (processed),
                INDEX idx_timestamp (timestamp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            conn.commit()
            logger.info("Loan table initialized (other tables must be created manually)")
            return conn

        except Error as e:
            logger.error(f"Error initializing database: {e}")
            if conn:
                conn.rollback()
            raise RuntimeError(f"Database initialization failed: {e}") from e
        finally:
            if cursor:
                cursor.close()

    def save_application(self, conn, input_data, prediction=None, probability=None):
        cursor = None
        try:
            if not conn or not conn.is_connected():
                conn = self.connect()
            cursor = conn.cursor()

            total_income = float(input_data.get('ApplicantIncome', 0)) + float(input_data.get('CoapplicantIncome', 0))
            loan_amount = float(input_data.get('LoanAmount', 0))
            loan_term = int(input_data.get('Loan_Amount_Term', 1))
            dependents_raw = input_data.get('Dependents', '0')
            dependents = float(str(dependents_raw).replace('3+', '3'))

            query = """
            INSERT INTO loan (
                loan_id, timestamp,
                gender, married, dependents, education, self_employed,
                applicant_income, coapplicant_income, loan_amount, loan_amount_term, 
                credit_history, property_area,
                total_income, loan_to_income_ratio, emi, balance_income,
                loan_amount_log, income_per_dependent, credit_history_income_interaction,
                prediction, probability, data_source, model_version
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                timestamp = VALUES(timestamp),
                prediction = VALUES(prediction),
                probability = VALUES(probability)
            """

            values = (
                str(input_data.get('Loan_ID')),
                datetime.now(),
                str(input_data.get('Gender')),
                str(input_data.get('Married')),
                str(dependents_raw),
                str(input_data.get('Education')),
                str(input_data.get('Self_Employed')),
                float(input_data.get('ApplicantIncome', 0)),
                float(input_data.get('CoapplicantIncome', 0)),
                float(input_data.get('LoanAmount', 0)),
                int(input_data.get('Loan_Amount_Term', 0)),
                str(input_data.get('Credit_History')),
                str(input_data.get('Property_Area')),
                float(total_income),
                float(loan_amount / (total_income + 1e-6) if total_income > 0 else 0),
                float(loan_amount / loan_term if loan_term > 0 else 0),
                float(total_income - ((loan_amount / loan_term) * 1000) if loan_term > 0 else total_income),
                float(np.log1p(loan_amount)),
                float(total_income / (dependents if dependents > 0 else 1)),
                float(total_income if input_data.get('Credit_History') == 'Good' else 0.0),
                int(prediction) if prediction is not None else None,
                float(probability) if probability is not None else None,
                'user_input',
                str(CONFIG.get("app", {}).get("content", {}).get("system_status", {}).get("model_version", "1.0"))
            )

            cursor.execute(query, values)
            conn.commit()
            logger.info(f"Saved application {input_data.get('Loan_ID')}")
            return True

        except Error as e:
            logger.error(f"Error saving application: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()

    def get_training_data(self, conn):
        try:
            if not conn or not conn.is_connected():
                conn = self.connect()

            query = """
            SELECT 
                gender, married, dependents, education, self_employed,
                applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                credit_history, property_area,
                total_income, loan_to_income_ratio, emi, balance_income,
                loan_amount_log, income_per_dependent, credit_history_income_interaction,
                CASE WHEN actual_status IS NOT NULL THEN actual_status 
                     WHEN prediction IS NOT NULL THEN IF(prediction=1, 'Y', 'N')
                     ELSE NULL END as loan_status
            FROM loan
            WHERE (actual_status IS NOT NULL OR prediction IS NOT NULL)
            AND processed = FALSE
            """
            return pd.read_sql(query, conn)
        except Error as e:
            logger.error(f"Error fetching training data: {e}")
            return pd.DataFrame()

    def mark_as_processed(self, conn, loan_ids):
        if not loan_ids:
            return
        cursor = None
        try:
            if not conn or not conn.is_connected():
                conn = self.connect()
            cursor = conn.cursor()
            placeholders = ','.join(['%s'] * len(loan_ids))
            query = f"UPDATE loan SET processed = TRUE WHERE loan_id IN ({placeholders})"
            cursor.execute(query, tuple(loan_ids))
            conn.commit()
            logger.info(f"Marked {len(loan_ids)} applications as processed")
        except Error as e:
            logger.error(f"Error marking applications as processed: {e}")
            if conn:
                conn.rollback()
        finally:
            if cursor:
                cursor.close()

    def update_actual_status(self, conn, loan_id, actual_status):
        cursor = None
        try:
            if not conn or not conn.is_connected():
                conn = self.connect()
            cursor = conn.cursor()
            query = "UPDATE loan SET actual_status = %s WHERE loan_id = %s"
            cursor.execute(query, (actual_status, loan_id))
            conn.commit()
            logger.info(f"Updated status for {loan_id} to {actual_status}")
            return True
        except Error as e:
            logger.error(f"Error updating status: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()


# Singleton instance
db = MySQLDatabase()

# Public interface
def init_db():
    return db.init_db()

def save_application(conn, input_data, prediction=None, probability=None):
    return db.save_application(conn, input_data, prediction, probability)

def get_training_data(conn):
    return db.get_training_data(conn)

def mark_as_processed(conn, loan_ids):
    return db.mark_as_processed(conn, loan_ids)

def update_actual_status(conn, loan_id, actual_status):
    return db.update_actual_status(conn, loan_id, actual_status)
