# view_counts.py
import sqlite3
from pathlib import Path

DB_PATH = Path("app/storage/entities/kb.sqlite")  # Update this path

def view_database_counts(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Count labels
    c.execute("SELECT COUNT(*) FROM labels")
    label_count = c.fetchone()[0]
    
    # Count entities
    c.execute("SELECT COUNT(*) FROM entities")
    entity_count = c.fetchone()[0]
    
    # Count seed entities
    c.execute("SELECT COUNT(*) FROM entities WHERE is_seed = 1")
    seed_entity_count = c.fetchone()[0]
    
    # Count entities by label
    c.execute("""
        SELECT l.label, COUNT(e.entity_text) as entity_count
        FROM entities e
        JOIN labels l ON e.label_id = l.label_id
        GROUP BY l.label
        ORDER BY entity_count DESC
    """)
    entities_by_label = c.fetchall()
    
    print("=== DATABASE SUMMARY ===")
    print(f"Total Labels: {label_count}")
    print(f"Total Entities: {entity_count}")
    print(f"Seed Entities: {seed_entity_count}")
    print(f"Non-Seed Entities: {entity_count - seed_entity_count}")
    
    print("\n=== ENTITIES BY LABEL ===")
    for label, count in entities_by_label:
        print(f"{label}: {count} entities")
    
    # Show label coverage info
    print("\n=== LABEL COVERAGE ===")
    c.execute("""
        SELECT l.label, lc.docs_with_label
        FROM label_coverage lc
        JOIN labels l ON lc.label_id = l.label_id
        ORDER BY lc.docs_with_label DESC
    """)
    coverage_data = c.fetchall()
    for label, docs_with_label in coverage_data:
        print(f"{label}: {docs_with_label} documents with this label")
    
    conn.close()

if __name__ == "__main__":
    view_database_counts(DB_PATH)