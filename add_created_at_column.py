#!/usr/bin/env python3
"""
Script to add created_at column to the recording table
"""
import os
import sqlite3
from datetime import datetime
import sys

# Get the database path
DB_PATH = os.path.join('instance', 'smart_nvr.db')

def add_created_at_column():
    """Add created_at column to the recording table"""
    print(f"Connecting to database at {DB_PATH}")
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(recording)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if 'created_at' in column_names:
            print("Column 'created_at' already exists in recording table.")
            conn.close()
            return
        
        # Add the column
        print("Adding 'created_at' column to recording table...")
        cursor.execute("ALTER TABLE recording ADD COLUMN created_at TIMESTAMP")
        
        # Update the column with current timestamp for existing records
        current_time = datetime.utcnow().isoformat()
        cursor.execute(f"UPDATE recording SET created_at = '{current_time}'")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print("Column 'created_at' added successfully.")
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    add_created_at_column()