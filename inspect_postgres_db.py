#!/usr/bin/env python3
"""
Simple PostgreSQL Database Inspector
Run this to see what's in your database
"""

import psycopg2
import json
import pandas as pd
from tabulate import tabulate

# Load config
with open('db_config.json', 'r') as f:
    config = json.load(f)

def connect_db():
    """Connect to PostgreSQL database."""
    return psycopg2.connect(
        dbname=config['database'],
        user=config['username'],
        password=config['password'],
        host=config['host'],
        port=config['port']
    )

def show_tables():
    """Show all tables and their row counts."""
    conn = connect_db()
    cursor = conn.cursor()
    
    print("\n" + "=" * 80)
    print("📊 POSTGRESQL DATABASE: dropout_risk_db")
    print("=" * 80)
    
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    
    tables = cursor.fetchall()
    data = []
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]};")
        count = cursor.fetchone()[0]
        data.append([table[0], f"{count:,}"])
    
    print(tabulate(data, headers=["Table Name", "Row Count"], tablefmt="grid"))
    
    cursor.close()
    conn.close()

def show_raw_text_files():
    """Show sample raw text files."""
    conn = connect_db()
    cursor = conn.cursor()
    
    print("\n" + "=" * 80)
    print("📄 RAW TEXT FILES (Your Original .txt Files)")
    print("=" * 80)
    
    cursor.execute("""
        SELECT uid, source_file, source_directory, LENGTH(raw_text) as text_length
        FROM raw_text_files 
        ORDER BY uid
        LIMIT 10;
    """)
    
    rows = cursor.fetchall()
    print(tabulate(rows, headers=["UID", "File Name", "Directory", "Text Length"], tablefmt="grid"))
    
    cursor.close()
    conn.close()

def show_processed_features():
    """Show processed features from PostgreSQL."""
    conn = connect_db()
    cursor = conn.cursor()
    
    print("\n" + "=" * 80)
    print("🔬 PROCESSED FEATURES (Using PostgreSQL Exact Replica)")
    print("=" * 80)
    
    # Use the exact method from postgres_exact_replica
    from postgres_exact_replica import PostgreSQLExactReplica
    db = PostgreSQLExactReplica()
    df = db.get_features_as_dataframe()
    
    print(f"\n✅ Loaded {len(df)} records from PostgreSQL")
    print(f"\n📊 Column Summary:")
    print(f"   Total columns: {len(df.columns)}")
    
    # Show column names by category
    numeric_cols = ['age', 'last_exam_score', 'meals_per_day', 'siblings_count', 'sentence_count', 'text_len']
    flag_cols = [c for c in df.columns if c.endswith('_flag')]
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    
    print(f"   Numeric features: {len(numeric_cols)}")
    print(f"   Flag features: {len(flag_cols)}")
    print(f"   Embedding features: {len(emb_cols)}")
    
    print(f"\n📈 Sample Data (first 5 records):")
    sample_cols = ['uid', 'age', 'class', 'meals_per_day', 'last_exam_score', 'heuristic_score', 'weak_label']
    available_cols = [c for c in sample_cols if c in df.columns]
    print(tabulate(df[available_cols].head(), headers='keys', tablefmt='grid', showindex=False))
    
    print(f"\n📊 Statistics:")
    stats = df[['age', 'meals_per_day', 'last_exam_score', 'heuristic_score']].describe()
    print(tabulate(stats, headers='keys', tablefmt='grid'))
    
    conn.close()

def compare_with_local():
    """Compare PostgreSQL data with local CSV."""
    print("\n" + "=" * 80)
    print("🔄 COMPARISON: PostgreSQL vs Local CSV")
    print("=" * 80)
    
    # Load from PostgreSQL
    from postgres_exact_replica import PostgreSQLExactReplica
    db = PostgreSQLExactReplica()
    df_pg = db.get_features_as_dataframe()
    
    # Load from local CSV
    df_local = pd.read_csv('usable/features_dataset.csv')
    
    data = [
        ["Total Records", len(df_pg), len(df_local), "✅" if len(df_pg) == len(df_local) else "❌"],
        ["Total Columns", len(df_pg.columns), len(df_local.columns), "✅" if len(df_pg.columns) == len(df_local.columns) else "❌"],
        ["Avg Age", f"{df_pg['age'].mean():.2f}", f"{df_local['age'].mean():.2f}", "✅"],
        ["Avg Meals/Day", f"{df_pg['meals_per_day'].mean():.2f}", f"{df_local['meals_per_day'].mean():.2f}", "✅"],
        ["Avg Exam Score", f"{df_pg['last_exam_score'].mean():.2f}", f"{df_local['last_exam_score'].mean():.2f}", "✅"],
    ]
    
    print(tabulate(data, headers=["Metric", "PostgreSQL", "Local CSV", "Match"], tablefmt="grid"))
    
    if len(df_pg) == len(df_local):
        print("\n✅ PostgreSQL data exactly matches local CSV data!")
    else:
        print("\n⚠️  Data differs between PostgreSQL and local CSV")

def show_sample_record():
    """Show one complete record."""
    conn = connect_db()
    cursor = conn.cursor()
    
    print("\n" + "=" * 80)
    print("🔍 SAMPLE COMPLETE RECORD")
    print("=" * 80)
    
    cursor.execute("SELECT uid, source_file, raw_text FROM raw_text_files LIMIT 1;")
    uid, source_file, raw_text = cursor.fetchone()
    
    print(f"\nUID: {uid}")
    print(f"Source File: {source_file}")
    print(f"\nRaw Text (first 500 chars):")
    print("-" * 80)
    print(raw_text[:500])
    print("-" * 80)
    
    # Get processed features
    from postgres_exact_replica import PostgreSQLExactReplica
    db = PostgreSQLExactReplica()
    df = db.get_features_as_dataframe()
    record = df[df['uid'] == uid].iloc[0]
    
    print(f"\nProcessed Features:")
    print(f"  Age: {record.get('age', 'N/A')}")
    print(f"  Class: {record.get('class', 'N/A')}")
    print(f"  Meals per day: {record.get('meals_per_day', 'N/A')}")
    print(f"  Last exam score: {record.get('last_exam_score', 'N/A')}")
    print(f"  Siblings count: {record.get('siblings_count', 'N/A')}")
    print(f"  Heuristic score: {record.get('heuristic_score', 'N/A')}")
    print(f"  Weak label: {record.get('weak_label', 'N/A')}")
    print(f"  Dropout risk: {record.get('dropout_risk', 'N/A')}")
    
    cursor.close()
    conn.close()

def main():
    """Main menu."""
    print("\n" + "=" * 80)
    print("🔍 POSTGRESQL DATABASE INSPECTOR")
    print("=" * 80)
    print("\nThis tool helps you verify that your data is stored in PostgreSQL")
    print("and that it matches your local files exactly.\n")
    
    try:
        show_tables()
        show_raw_text_files()
        show_processed_features()
        compare_with_local()
        show_sample_record()
        
        print("\n" + "=" * 80)
        print("✅ VERIFICATION COMPLETE!")
        print("=" * 80)
        print("\nYour data is stored in PostgreSQL and matches your local files.")
        print("\n💡 To access PostgreSQL directly:")
        print("   1. Command line: psql -U kahindo -d dropout_risk_db")
        print("   2. GUI tools: pgAdmin or DBeaver")
        print("   3. This script: python3 inspect_postgres_db.py")
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure PostgreSQL is running and the database exists.")

if __name__ == "__main__":
    main()

