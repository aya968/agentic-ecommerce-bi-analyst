import sqlite3
import re
from pathlib import Path
from uuid import uuid4

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
    def _quote_identifier(identifier):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier or ""):
            raise ValueError("Invalid SQLite table name.")
        return f'"{identifier}"'

    @staticmethod
    def normalize_column_name(name):
        name = str(name).strip()
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        name = name.lower()
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

        original_column_names = list(df.columns)
        normalized_column_names = self.make_unique_columns(df.columns)
        original_by_normalized = dict(
            zip(normalized_column_names, original_column_names)
        )
        df.columns = normalized_column_names

        # Remove fully empty rows and columns
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")

        # Remove duplicate rows
        duplicates_removed = int(df.duplicated().sum())
        df = df.drop_duplicates()

        # Trim text values
        for column in df.columns:
            dtype = df[column].dtype
            if not (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
            ):
                continue
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
            "columns": list(df.columns),
            "column_name_mapping": [
                {
                    "original": str(original_by_normalized[column]),
                    "cleaned": str(column)
                }
                for column in df.columns
            ]
        }

        return df, report

    def prepare_file(self, file_path):
        df = self.read_file(file_path)
        cleaned_df, report = self.clean_dataframe(df)

        if cleaned_df.empty:
            raise ValueError(
                "The uploaded file contains no usable rows."
            )

        return cleaned_df, report

    def save_dataframe_to_sqlite(
        self,
        dataframe,
        table_name="ecommerce_data",
        report=None
    ):
        self._quote_identifier(table_name)
        if dataframe.empty:
            raise ValueError(
                "The uploaded file contains no usable rows."
            )

        staging_table = f"_staging_{uuid4().hex}"
        quoted_target = self._quote_identifier(table_name)
        quoted_staging = self._quote_identifier(staging_table)
        connection = self.get_connection()
        try:
            dataframe.to_sql(
                staging_table,
                connection,
                if_exists="fail",
                index=False
            )
            connection.execute("BEGIN")
            target_exists = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name=?",
                (table_name,)
            ).fetchone() is not None
            backup_table = f"_backup_{uuid4().hex}"

            if target_exists:
                connection.execute(
                    f"ALTER TABLE {quoted_target} "
                    f"RENAME TO {self._quote_identifier(backup_table)}"
                )

            connection.execute(
                f"ALTER TABLE {quoted_staging} RENAME TO {quoted_target}"
            )

            if target_exists:
                connection.execute(
                    f"DROP TABLE {self._quote_identifier(backup_table)}"
                )
            connection.commit()
        except Exception:
            connection.rollback()
            staging_exists = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name=?",
                (staging_table,)
            ).fetchone() is not None
            if staging_exists:
                connection.execute(f"DROP TABLE {quoted_staging}")
                connection.commit()
            raise
        finally:
            connection.close()

        result_report = dict(report or {})
        result_report["table_name"] = table_name
        result_report["loaded_rows"] = len(dataframe)

        return result_report

    def load_file_to_sqlite(
        self,
        file_path,
        table_name="ecommerce_data"
    ):
        dataframe, report = self.prepare_file(file_path)
        return self.save_dataframe_to_sqlite(
            dataframe,
            table_name=table_name,
            report=report
        )

    def get_table_names(self):
        connection = self.get_connection()
        try:
            result = pd.read_sql_query(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                AND name NOT LIKE 'sqlite_%'
                """,
                connection
            )
        finally:
            connection.close()

        return result["name"].tolist()

    def get_dataframe(self, table_name="ecommerce_data"):
        quoted_table = self._quote_identifier(table_name)
        connection = self.get_connection()
        try:
            return pd.read_sql_query(
                f"SELECT * FROM {quoted_table}",
                connection
            )
        finally:
            connection.close()

    def get_schema(self, table_name="ecommerce_data"):
        quoted_table = self._quote_identifier(table_name)
        connection = self.get_connection()
        try:
            cursor = connection.cursor()

            cursor.execute(
                f"PRAGMA table_info({quoted_table})"
            )

            rows = cursor.fetchall()
        finally:
            connection.close()

        return [
            {
                "name": row[1],
                "type": row[2]
            }
            for row in rows
        ]
        