# setup_database.py
from database import MySQLDatabase
import mysql.connector
from mysql.connector import Error

def create_database():
    """Create the loan database if it doesn't exist"""
    try:
        # Connect without specifying database
        conn = mysql.connector.connect(
            host="localhost",
            user="root",   
            password="1111"
        )
        
        if conn.is_connected():
            cursor = conn.cursor()
            cursor.execute("CREATE DATABASE IF NOT EXISTS loan CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print("Database 'loan' created successfully")
            
    except Error as e:
        print(f"Error creating database: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def initialize_tables():
    """Initialize all tables by connecting to the database"""
    db = MySQLDatabase()
    db.init_db()
    print("Database tables initialized successfully")

if __name__ == "__main__":
    create_database()
    initialize_tables()