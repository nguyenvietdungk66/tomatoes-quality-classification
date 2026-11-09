import sqlite3
from pathlib import Path


class DatabaseManager:
    def __init__(self, database_path=None):
        project_root = Path(__file__).resolve().parents[2]
        if database_path is None:
            database_path = project_root / "data" / "inspection_history.db"

        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_database()

    def get_connection(self):
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize_database(self):
        connection = self.get_connection()
        try:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS inspection_sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    total_inspected INTEGER NOT NULL DEFAULT 0,
                    total_fresh INTEGER NOT NULL DEFAULT 0,
                    total_rotten INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS inspection_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    inspection_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id)
                        REFERENCES inspection_sessions(session_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_results_session
                    ON inspection_results(session_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_start_time
                    ON inspection_sessions(start_time);
            """)

            # Backward-compatible migration for databases created by older versions.
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(inspection_sessions)"
                ).fetchall()
            }
            migrations = {
                "session_name": "ALTER TABLE inspection_sessions ADD COLUMN session_name TEXT NOT NULL DEFAULT 'Unnamed session'",
                "status": "ALTER TABLE inspection_sessions ADD COLUMN status TEXT NOT NULL DEFAULT 'PAUSED'",
                "checkpoint_json": "ALTER TABLE inspection_sessions ADD COLUMN checkpoint_json TEXT",
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)

            connection.commit()
        finally:
            connection.close()
