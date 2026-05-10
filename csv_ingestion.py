"""
CSV Text Extraction Module

This module converts CSV rows into searchable text while preserving
column context for RAG retrieval.
"""

import csv
import os


def get_csv_text(file_path):
    """
    Extract text from a CSV file by converting rows into labeled text.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        str: Text representation of the CSV rows
    """
    if not os.path.exists(file_path):
        return f"❌ ERROR: File not found - {file_path}"

    if not file_path.lower().endswith(".csv"):
        return f"❌ ERROR: File is not a CSV - {file_path}"

    try:
        with open(file_path, mode="r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                return "❌ ERROR: CSV file is missing headers."

            rows = []
            for index, row in enumerate(reader, start=1):
                fields = []
                for field, raw_value in row.items():
                    value = "" if raw_value is None else str(raw_value).strip()
                    if value:
                        fields.append(f"{field}: {value}")
                if fields:
                    rows.append(f"Row {index}: " + "; ".join(fields))

            if not rows:
                return "❌ ERROR: CSV file contains no data rows."

            return "\n".join(rows)
    except Exception as e:
        return f"❌ ERROR: Failed to read CSV - {str(e)}"


if __name__ == "__main__":
    example_path = "sample.csv"
    print(get_csv_text(example_path))
