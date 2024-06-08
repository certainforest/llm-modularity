import pandas as pd
import sqlite3
import os

class SQLiteConn:
    """
    Helper for handling SQlite connections
    """
    def __init__(self, sqlite_file: str):
        self.sqlite_file = sqlite_file
        self._create_database_if_not_exists()

    def _create_database_if_not_exists(self):
        if not os.path.exists(self.sqlite_file):
            print('Creating database ' + os.path.abspath(self.sqlite_file))
            with sqlite3.connect(self.sqlite_file) as conn:
                pass

    def execute(self, query: str):
        with sqlite3.connect(self.sqlite_file) as conn:
            cur = conn.cursor()
            cur.execute(query)

    def get_query(self, query: str) -> pd.DataFrame:
        with sqlite3.connect(self.sqlite_file) as conn:
            return pd.read_sql_query(query, conn)

    def write_df(self, tablename: str, df: pd.DataFrame):
        with sqlite3.connect(self.sqlite_file) as conn:
            df.to_sql(tablename, conn, if_exists = 'append', index=False)