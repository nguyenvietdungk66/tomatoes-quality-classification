
import json


class InspectionRepository:

    def __init__(self, database_manager):
        self.database_manager = database_manager

    # ==========================================
    # GENERATE UNIQUE SESSION NAME
    # ==========================================

    def _generate_unique_session_name(self, cursor, session_name):

        base_name = str(
            session_name or "Unnamed session"
        ).strip()

        if not base_name:
            base_name = "Unnamed session"

        # Check whether the original name already exists.
        cursor.execute("""
            SELECT 1
            FROM inspection_sessions
            WHERE session_name = ? COLLATE NOCASE
            LIMIT 1
        """, (base_name,))

        if cursor.fetchone() is None:
            return base_name

        # Generate Windows-style suffix:
        # Name (1), Name (2), Name (3), ...

        index = 1

        while True:

            candidate_name = f"{base_name} ({index})"

            cursor.execute("""
                SELECT 1
                FROM inspection_sessions
                WHERE session_name = ? COLLATE NOCASE
                LIMIT 1
            """, (candidate_name,))

            if cursor.fetchone() is None:
                return candidate_name

            index += 1

    # ==========================================
    # SAVE SESSION
    # ==========================================

    def save_session(
        self,
        start_time,
        end_time,
        total_inspected,
        total_fresh,
        total_rotten,
        results,
        session_name="Unnamed session",
        checkpoint=None,
        session_id=None,
        status="PAUSED"
    ):

        connection = self.database_manager.get_connection()

        checkpoint_json = (
            json.dumps(checkpoint, ensure_ascii=False)
            if checkpoint is not None
            else None
        )

        try:

            cursor = connection.cursor()

            if session_id is None:

                # ----------------------------------
                # NEW SESSION
                # ----------------------------------

                # Acquire write lock before checking
                # the name to reduce duplicate-name
                # conflicts between simultaneous saves.

                cursor.execute("BEGIN IMMEDIATE")

                unique_name = self._generate_unique_session_name(
                    cursor,
                    session_name
                )

                cursor.execute("""
                    INSERT INTO inspection_sessions (
                        session_name,
                        start_time,
                        end_time,
                        total_inspected,
                        total_fresh,
                        total_rotten,
                        status,
                        checkpoint_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    unique_name,
                    start_time,
                    end_time,
                    total_inspected,
                    total_fresh,
                    total_rotten,
                    status,
                    checkpoint_json
                ))

                session_id = cursor.lastrowid

            else:

                # ----------------------------------
                # UPDATE EXISTING SESSION
                # ----------------------------------

                # Keep the original session name.
                # Resuming a session must not create
                # another name or another session.

                cursor.execute("""
                    UPDATE inspection_sessions
                    SET
                        start_time = ?,
                        end_time = ?,
                        total_inspected = ?,
                        total_fresh = ?,
                        total_rotten = ?,
                        status = ?,
                        checkpoint_json = ?
                    WHERE session_id = ?
                """, (
                    start_time,
                    end_time,
                    total_inspected,
                    total_fresh,
                    total_rotten,
                    status,
                    checkpoint_json,
                    session_id
                ))

                if cursor.rowcount == 0:

                    raise ValueError(
                        f"Session {session_id} does not exist"
                    )

                # Replace previous results with
                # the updated session results.

                cursor.execute("""
                    DELETE FROM inspection_results
                    WHERE session_id = ?
                """, (session_id,))

            # ----------------------------------
            # SAVE INSPECTION RESULTS
            # ----------------------------------

            result_rows = [
                (
                    session_id,
                    result["inspection_id"],
                    result["class_name"],
                    result["confidence"],
                    result["timestamp"]
                )
                for result in results
            ]

            if result_rows:

                cursor.executemany("""
                    INSERT INTO inspection_results (
                        session_id,
                        inspection_id,
                        class_name,
                        confidence,
                        timestamp
                    )
                    VALUES (?, ?, ?, ?, ?)
                """, result_rows)

            connection.commit()

            return session_id

        except Exception:

            connection.rollback()
            raise

        finally:

            connection.close()

    # ==========================================
    # GET ALL SESSIONS
    # ==========================================

    def get_sessions(self):

        connection = self.database_manager.get_connection()

        try:

            rows = connection.execute("""
                SELECT
                    session_id,
                    session_name,
                    start_time,
                    end_time,
                    total_inspected,
                    total_fresh,
                    total_rotten,
                    status
                FROM inspection_sessions
                ORDER BY start_time DESC, session_id DESC
            """).fetchall()

            return [
                dict(row)
                for row in rows
            ]

        finally:

            connection.close()

    # ==========================================
    # GET SESSION
    # ==========================================

    def get_session(self, session_id):

        connection = self.database_manager.get_connection()

        try:

            row = connection.execute(
                """
                SELECT *
                FROM inspection_sessions
                WHERE session_id = ?
                """,
                (session_id,)
            ).fetchone()

            if row is None:
                return None

            data = dict(row)

            raw_checkpoint = data.pop(
                "checkpoint_json",
                None
            )

            data["checkpoint"] = (
                json.loads(raw_checkpoint)
                if raw_checkpoint
                else None
            )

            return data

        finally:

            connection.close()

    # ==========================================
    # GET SESSION RESULTS
    # ==========================================

    def get_session_results(self, session_id):

        connection = self.database_manager.get_connection()

        try:

            rows = connection.execute("""
                SELECT
                    result_id,
                    session_id,
                    inspection_id,
                    class_name,
                    confidence,
                    timestamp
                FROM inspection_results
                WHERE session_id = ?
                ORDER BY result_id ASC
            """, (session_id,)).fetchall()

            return [
                dict(row)
                for row in rows
            ]

        finally:

            connection.close()

    # ==========================================
    # DELETE SESSION
    # ==========================================

    def delete_session(self, session_id):

        connection = self.database_manager.get_connection()

        try:

            cursor = connection.execute(
                """
                DELETE FROM inspection_sessions
                WHERE session_id = ?
                """,
                (session_id,)
            )

            connection.commit()

            return cursor.rowcount > 0

        except Exception:

            connection.rollback()
            raise

        finally:

            connection.close()

    # ==========================================
    # DASHBOARD STATISTICS
    # ==========================================

    def get_dashboard_statistics(self):

        """
        Get cumulative statistics from all saved sessions.
        Each session is counted only once.
        """

        connection = self.database_manager.get_connection()

        try:

            row = connection.execute("""
                SELECT
                    COUNT(*) AS total_sessions,
                    COALESCE(SUM(total_inspected), 0) AS total_inspected,
                    COALESCE(SUM(total_fresh), 0) AS total_fresh,
                    COALESCE(SUM(total_rotten), 0) AS total_rotten
                FROM inspection_sessions
            """).fetchone()

            return dict(row)

        finally:

            connection.close()

    # ==========================================
    # DAILY STATISTICS
    # ==========================================

    def get_daily_statistics(self, days=7):

        """
        Get inspection volume grouped by session start date.
        Returns the last N calendar days, including
        days with zero inspections.
        """

        from datetime import date, timedelta

        days = max(1, int(days))

        today = date.today()

        start_date = today - timedelta(
            days=days - 1
        )

        connection = self.database_manager.get_connection()

        try:

            rows = connection.execute("""
                SELECT
                    date(start_time) AS inspection_date,
                    COALESCE(SUM(total_inspected), 0) AS total
                FROM inspection_sessions
                WHERE date(start_time) >= ?
                  AND date(start_time) <= ?
                GROUP BY date(start_time)
                ORDER BY inspection_date ASC
            """, (
                start_date.isoformat(),
                today.isoformat()
            )).fetchall()

            daily_map = {
                row["inspection_date"]: row["total"]
                for row in rows
                if row["inspection_date"] is not None
            }

            result = []

            for offset in range(days):

                current_date = start_date + timedelta(
                    days=offset
                )

                date_key = current_date.isoformat()

                result.append({
                    "date": date_key,
                    "total": int(
                        daily_map.get(date_key, 0)
                    )
                })

            return result

        finally:

            connection.close()
