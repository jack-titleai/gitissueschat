#!/usr/bin/env python3
"""
Migration script to add new columns to the database tables.
"""

import sqlite3
import os
import argparse
from datetime import datetime


def migrate_database(db_path):
    """
    Migrate the database to add new columns to tables.
    """
    print(f"Migrating database at {db_path}...")

    # Check if the database file exists
    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} does not exist.")
        return False

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Get current UTC timestamp
        current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Add added_at column to repositories table
        cursor.execute("PRAGMA table_info(repositories)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]

        if "added_at" not in column_names:
            print("Adding added_at column to repositories table...")
            cursor.execute("ALTER TABLE repositories ADD COLUMN added_at TIMESTAMP")
            cursor.execute("UPDATE repositories SET added_at = ?", (current_time,))
            print("Successfully added added_at column to repositories table.")
        else:
            print("The added_at column already exists in repositories table.")

        # Add added_at column to issues table
        cursor.execute("PRAGMA table_info(issues)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]

        if "added_at" not in column_names:
            print("Adding added_at column to issues table...")
            cursor.execute("ALTER TABLE issues ADD COLUMN added_at TIMESTAMP")
            cursor.execute("UPDATE issues SET added_at = ?", (current_time,))
            print("Successfully added added_at column to issues table.")
        else:
            print("The added_at column already exists in issues table.")

        # Add added_at column to comments table
        cursor.execute("PRAGMA table_info(comments)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]

        if "added_at" not in column_names:
            print("Adding added_at column to comments table...")
            cursor.execute("ALTER TABLE comments ADD COLUMN added_at TIMESTAMP")
            cursor.execute("UPDATE comments SET added_at = ?", (current_time,))
            print("Successfully added added_at column to comments table.")
        else:
            print("The added_at column already exists in comments table.")

        # Check if the redundant_issues_count column already exists in api_logs
        cursor.execute("PRAGMA table_info(api_logs)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]

        if "redundant_issues_count" not in column_names:
            print("Adding redundant_issues_count column to api_logs table...")
            cursor.execute(
                "ALTER TABLE api_logs ADD COLUMN redundant_issues_count INTEGER DEFAULT 0"
            )
            print("Successfully added redundant_issues_count column to api_logs table.")
        else:
            print("The redundant_issues_count column already exists in api_logs table.")

        conn.commit()
        conn.close()
        return True

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        conn.close()
        return False


def main():
    """Main function to run the migration script."""
    parser = argparse.ArgumentParser(description="Migrate the database to add new columns.")
    parser.add_argument(
        "--db-path", default="github_issues.db", help="Path to the SQLite database file"
    )

    args = parser.parse_args()

    if migrate_database(args.db_path):
        print("Migration completed successfully.")
    else:
        print("Migration failed.")
        exit(1)


if __name__ == "__main__":
    main()
