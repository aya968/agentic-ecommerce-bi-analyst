import sqlite3
import re
from pathlib import Path

import pandas as pd

from app.config import DB_PATH


class DatabaseManager:

    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    @staticmethod
    def normalize_column_name(name):
        name = str(name).strip().lower()
        name = re.sub(r"[^\w]+", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")

        if not name:
            name = "unnamed_column"

        if name[0].isdigit():
            name = f"col_{name}"

        return name

    def make_unique_columns(self, columns):
        result = []
        counts = {}

        for column in columns:
            base = self.normalize_column_name(column)

            if base not in counts:
                counts[base] = 0
                new_name = base
            else:
                counts[base] += 1
                new_name = f"{base}_{counts[base]}"

            result.append(new_name)

        return result

    def read_file(self, file_path):
        extension = Path(file_path).suffix.lower()

        if extension == ".csv":
            try:
                return pd.read_csv(file_path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                return pd.read_csv(file_path, encoding="latin-1")

        if extension in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)

        raise ValueError(
            "Unsupported file. Please upload CSV or Excel."
        )

    def clean_dataframe(self, df):
        original_rows = len(df)
        original_columns = len(df.columns)

        df = df.copy()

        df.columns = self.make_unique_columns(df.columns)

        # Remove fully empty rows and columns
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")

        # Remove duplicate rows
        duplicates_removed = int(df.duplicated().sum())
        df = df.drop_duplicates()

        # Trim text values
        for column in df.select_dtypes(include=["object"]).columns:
            df[column] = df[column].apply(
                lambda value: value.strip()
                if isinstance(value, str)
                else value
            )

        # Attempt date conversion for likely date columns
        date_columns = []

        for column in df.columns:
            if any(
                word in column
                for word in ["date", "time", "created", "order_at"]
            ):
                converted = pd.to_datetime(
                    df[column],
                    errors="coerce"
                )

                valid_ratio = converted.notna().mean()

                if valid_ratio >= 0.6:
                    df[column] = converted
                    date_columns.append(column)

        report = {
            "original_rows": original_rows,
            "cleaned_rows": len(df),
            "removed_rows": original_rows - len(df),
            "duplicates_removed": duplicates_removed,
            "original_columns": original_columns,
            "cleaned_columns": len(df.columns),
            "date_columns_detected": date_columns,
            "columns": list(df.columns)
        }

        return df, report

    def load_file_to_sqlite(
        self,
        file_path,
        table_name="ecommerce_data"
    ):
        df = self.read_file(file_path)

        cleaned_df, report = self.clean_dataframe(df)

        if cleaned_df.empty:
            raise ValueError(
                "The uploaded file contains no usable rows."
            )

        with self.get_connection() as connection:
            cleaned_df.to_sql(
                table_name,
                connection,
                if_exists="replace",
                index=False
            )

        report["table_name"] = table_name
        report["loaded_rows"] = len(cleaned_df)

        return report

    def get_table_names(self):
        with self.get_connection() as connection:
            result = pd.read_sql_query(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                AND name NOT LIKE 'sqlite_%'
                """,
                connection
            )

        return result["name"].tolist()

    def get_dataframe(self, table_name="ecommerce_data"):
        with self.get_connection() as connection:
            return pd.read_sql_query(
                f'SELECT * FROM "{table_name}"',
                connection
            )

    def get_schema(self, table_name="ecommerce_data"):
        with self.get_connection() as connection:
            cursor = connection.cursor()

            cursor.execute(
                f'PRAGMA table_info("{table_name}")'
            )

            rows = cursor.fetchall()

        return [
            {
                "name": row[1],
                "type": row[2]
            }
            for row in rows
        ]
        