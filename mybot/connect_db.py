"""
CSV Import Script for Financial Transactions
Loads transaction data from CSV into PostgreSQL database
"""
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

def main():
    # Load environment variables
    load_dotenv()

    # Get DB credentials from .env
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "finance_db")
    DB_DRIVER = os.getenv("DB_DRIVER", "postgresql+psycopg2")
    TABLE_NAME = os.getenv("EXPENSES_TABLE_NAME", "transactions")

    # Validate required environment variables
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("❌ ERROR: Missing required database environment variables")
        print("   Please check your .env file contains:")
        print("   - DB_USER")
        print("   - DB_PASSWORD")
        print("   - DB_HOST")
        print("   - DB_PORT")
        print("   - DB_NAME")
        sys.exit(1)

    # CSV file path - configurable via environment or command line
    csv_path = os.getenv("CSV_PATH", "./data/transactions.csv")
    
    # Allow command line override
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    # Validate file exists
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"❌ ERROR: CSV file not found: {csv_path}")
        print(f"   Please ensure the file exists or set CSV_PATH in .env")
        print(f"   Usage: python import_csv.py [path/to/file.csv]")
        sys.exit(1)

    print(f"📂 Loading CSV from: {csv_path}")
    
    try:
        # Load CSV into DataFrame
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} records from CSV")
        print(f"  Columns: {', '.join(df.columns.tolist())}")

        # Create database engine
        db_url = f"{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        print(f"✓ Connected to database: {DB_NAME}")

        # Import CSV into table
        print(f"📥 Importing to table '{TABLE_NAME}'...")
        df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
        
        print(f"✅ CSV imported successfully!")
        print(f"   {len(df)} records loaded into '{TABLE_NAME}' table")
        
        # Show sample of data
        print(f"\n📊 Sample data (first 3 rows):")
        print(df.head(3).to_string())

    except pd.errors.EmptyDataError:
        print(f"❌ ERROR: CSV file is empty")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"❌ ERROR: Failed to parse CSV file")
        print(f"   {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Failed to import CSV")
        print(f"   {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()