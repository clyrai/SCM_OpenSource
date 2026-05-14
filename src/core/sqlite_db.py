"""
SleepAI SQLite Database (for testing/development without PostgreSQL)
"""
import sqlite3
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import os

from ..core.config import DATA_DIR
from .time_utils import ensure_utc, utc_isoformat

DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "sleepai.db"


def set_db_path(path) -> None:
    """
    Override the SQLite DB path at runtime and reset the cached connection
    + singleton instance so the next get_memory() call uses the new path.
    Used by tests to work against a temp DB without polluting production data.
    """
    global DB_PATH, _memory
    from pathlib import Path as _Path
    DB_PATH = _Path(path)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _memory = None
    # Re-initialize schema on the new path
    init_db()


def get_connection():
    """Get SQLite connection"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize SQLite database schema"""
    conn = get_connection()
    cursor = conn.cursor()

    # Concepts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            description TEXT NOT NULL,
            embedding TEXT,
            novelty REAL DEFAULT 0.5,
            emotional REAL DEFAULT 0.0,
            task_relevance REAL DEFAULT 0.5,
            repetition REAL DEFAULT 0.5,
            importance_score REAL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            last_accessed TEXT NOT NULL,
            access_count INTEGER DEFAULT 0,
            strength REAL DEFAULT 1.0,
            state TEXT DEFAULT 'active',
            version_root TEXT,
            version_parent TEXT,
            valid_from TEXT,
            valid_to TEXT,
            is_current_version INTEGER DEFAULT 1,
            context_tags TEXT
        )
    """)

    # Relations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS relations (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_id TEXT NOT NULL,
            strength REAL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            bidirectional INTEGER DEFAULT 0
        )
    """)

    # Episodes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            concept_ids TEXT,
            raw_content TEXT NOT NULL,
            context TEXT,
            importance_json TEXT,
            state TEXT DEFAULT 'active',
            source TEXT DEFAULT 'user'
        )
    """)

    # Sleep cycles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sleep_cycles (
            id TEXT PRIMARY KEY,
            start_time TEXT NOT NULL,
            end_time TEXT,
            nrem_duration REAL DEFAULT 0.0,
            rem_duration REAL DEFAULT 0.0,
            memories_consolidated INTEGER DEFAULT 0,
            memories_forgotten INTEGER DEFAULT 0,
            dreams_json TEXT
        )
    """)

    # Session metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_meta (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            last_active TEXT NOT NULL,
            message_count INTEGER DEFAULT 0,
            total_concepts INTEGER DEFAULT 0,
            total_sleeps INTEGER DEFAULT 0,
            ltm_json TEXT
        )
    """)

    # Per-user circadian sleep schedule (v0.7.7+).
    # Each row: a user's preferred nightly consolidation window in their
    # local timezone. The MCP sweeper uses these to fire deep-sleep at the
    # right wall-clock moment for each user, instead of after a fixed
    # idle gap.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_sleep_config (
            user_id TEXT PRIMARY KEY,
            timezone TEXT NOT NULL DEFAULT 'UTC',
            sleep_start TEXT NOT NULL DEFAULT '23:00',
            sleep_end TEXT NOT NULL DEFAULT '07:00',
            enabled INTEGER NOT NULL DEFAULT 1,
            last_sleep_at TEXT,
            updated_at TEXT NOT NULL
        )
    """)

    # ── SCM Cloud auth (v0.7.8+) ─────────────────────────────────────────
    # accounts:    one row per cloud customer (their email is the identity)
    # api_keys:    one row per issued key. Multiple keys per account is OK
    #              (revoke individually, rotate, scope to specific actions)
    # The "memory user_id" used by the existing /v1/memories/* API is
    # ALWAYS scoped under an account: account.id + namespace ⇒ user_id.
    # That keeps tenancy enforced even when the caller passes their own
    # user_id parameter.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            tier TEXT NOT NULL DEFAULT 'free',
            byok_llm_provider TEXT,
            byok_llm_api_key_enc TEXT,
            byok_llm_base_url TEXT,
            byok_llm_model TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            key_prefix TEXT NOT NULL,
            key_hash TEXT NOT NULL,
            label TEXT,
            scopes TEXT NOT NULL DEFAULT 'memories:rw',
            rate_limit_per_min INTEGER NOT NULL DEFAULT 60,
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            revoked_at TEXT,
            FOREIGN KEY(account_id) REFERENCES accounts(id)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_api_keys_account ON api_keys(account_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)"
    )

    conn.commit()
    _ensure_concept_version_columns(conn)
    _ensure_episode_session_columns(conn)
    conn.close()
    print(f"Database initialized at {DB_PATH}")


def _ensure_concept_version_columns(conn):
    """Add Phase 5 version columns when upgrading an existing SQLite DB."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(concepts)")
    existing = {row[1] for row in cursor.fetchall()}

    columns = {
        "version_root": "TEXT",
        "version_parent": "TEXT",
        "valid_from": "TEXT",
        "valid_to": "TEXT",
        "is_current_version": "INTEGER DEFAULT 1",
        "context_tags": "TEXT",
    }

    for column, ddl in columns.items():
        if column not in existing:
            cursor.execute(f"ALTER TABLE concepts ADD COLUMN {column} {ddl}")

    conn.commit()


def _ensure_episode_session_columns(conn):
    """
    Phase 7 migration: tag every episode with the session that produced it,
    so cross-session sleep cycles can pull recent prior-session episodes
    without leaking unrelated user data.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(episodes)")
    existing = {row[1] for row in cursor.fetchall()}

    columns = {
        "session_id": "TEXT",
    }
    for column, ddl in columns.items():
        if column not in existing:
            cursor.execute(f"ALTER TABLE episodes ADD COLUMN {column} {ddl}")

    # Index for fast "most recent N episodes for session X" lookups.
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_episodes_session_ts "
        "ON episodes(session_id, timestamp)"
    )
    conn.commit()


def _parse_datetime(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    try:
        return ensure_utc(value)
    except Exception:
        return None


class SQLiteMemory:
    """SQLite-based memory storage for testing"""

    def __init__(self):
        init_db()

    def save_concept(self, concept) -> bool:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO concepts
                (id, type, description, embedding, novelty, emotional, task_relevance,
                 repetition, importance_score, created_at, last_accessed, access_count, strength,
                 state, version_root, version_parent, valid_from, valid_to, is_current_version, context_tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                concept.id,
                concept.type.value if hasattr(concept.type, 'value') else concept.type,
                concept.description,
                json.dumps(concept.embedding) if concept.embedding else None,
                concept.importance.novelty,
                concept.importance.emotional,
                concept.importance.task_relevance,
                concept.importance.repetition,
                concept.importance.overall,
                concept.created_at.isoformat(),
                concept.last_accessed.isoformat(),
                concept.access_count,
                concept.strength,
                concept.state.value if hasattr(concept.state, 'value') else concept.state,
                getattr(concept, "version_root", None) or concept.id,
                getattr(concept, "version_parent", None),
                getattr(concept, "valid_from", None).isoformat() if getattr(concept, "valid_from", None) else None,
                getattr(concept, "valid_to", None).isoformat() if getattr(concept, "valid_to", None) else None,
                1 if getattr(concept, "is_current_version", True) else 0,
                json.dumps(getattr(concept, "context_tags", {}) or {}, default=str),
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving concept: {e}")
            return False
        finally:
            conn.close()

    def get_concept(self, concept_id: str):
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM concepts WHERE id = ?", (concept_id,))
            row = cursor.fetchone()

            if row:
                return {
                    'id': row['id'],
                    'type': row['type'],
                    'description': row['description'],
                    'embedding': json.loads(row['embedding']) if row['embedding'] else None,
                    'novelty': row['novelty'],
                    'emotional': row['emotional'],
                    'task_relevance': row['task_relevance'],
                    'repetition': row['repetition'],
                    'importance': row['importance_score'],
                    'created_at': row['created_at'],
                    'last_accessed': row['last_accessed'],
                    'access_count': row['access_count'],
                    'strength': row['strength'],
                    'state': row['state'],
                    'version_root': row['version_root'] if 'version_root' in row.keys() else None,
                    'version_parent': row['version_parent'] if 'version_parent' in row.keys() else None,
                    'valid_from': row['valid_from'] if 'valid_from' in row.keys() else None,
                    'valid_to': row['valid_to'] if 'valid_to' in row.keys() else None,
                    'is_current_version': bool(row['is_current_version']) if 'is_current_version' in row.keys() else True,
                    'context_tags': json.loads(row['context_tags']) if 'context_tags' in row.keys() and row['context_tags'] else {},
                }
            return None
        finally:
            conn.close()

    def search_concepts(self, query: str, limit: int = 5) -> List[Dict]:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM concepts
                WHERE description LIKE ?
                  AND state = 'active'
                  AND COALESCE(is_current_version, 1) = 1
                ORDER BY importance_score DESC
                LIMIT ?
            """, (f"%{query}%", limit))
            rows = cursor.fetchall()

            return [
                {
                    'id': row['id'],
                    'type': row['type'],
                    'description': row['description'],
                    'importance': row['importance_score'],
                    'created_at': row['created_at']
                }
                for row in rows
            ]
        finally:
            conn.close()

    def save_episode(self, episode, session_id: Optional[str] = None) -> bool:
        """
        Persist an episode. Phase 7: caller passes session_id so cross-session
        sleep can later filter by session. session_id may also be present in
        episode.context["session_id"] (set by ChatEngine ingestion); we accept
        both and prefer the explicit argument.
        """
        if session_id is None and isinstance(episode.context, dict):
            session_id = episode.context.get("session_id")

        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO episodes
                (id, timestamp, concept_ids, raw_content, context, importance_json, state, source, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.id,
                episode.timestamp.isoformat(),
                json.dumps(episode.concept_ids),
                episode.raw_content,
                json.dumps(episode.context),
                json.dumps(episode.importance.model_dump()),
                episode.state.value if hasattr(episode.state, 'value') else episode.state,
                episode.source,
                session_id,
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving episode: {e}")
            return False
        finally:
            conn.close()

    def get_recent_episodes_for_sessions(
        self,
        session_ids: Optional[List[str]] = None,
        since_iso: Optional[str] = None,
        max_per_session: int = 50,
        max_total: int = 200,
    ) -> List[Dict]:
        """
        Phase 7: pull recent episodes across one or more sessions.

        Used by the cross-session memory pool to give sleep cycles a rolling
        window of prior-session experience to consolidate.

        Args:
            session_ids: filter to these sessions. If None, all sessions.
            since_iso: only episodes with timestamp >= this ISO string.
            max_per_session: cap per session to prevent one chatty session
                from dominating the pool.
            max_total: hard cap on returned rows.
        """
        conn = get_connection()
        try:
            cursor = conn.cursor()
            results: List[Dict] = []
            if session_ids is None:
                # All sessions
                query = "SELECT * FROM episodes WHERE state = 'active'"
                params: list = []
                if since_iso:
                    query += " AND timestamp >= ?"
                    params.append(since_iso)
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(int(max_total))
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                results = [dict(r) for r in rows]
            else:
                # Per-session caps for fairness
                for sid in session_ids:
                    query = (
                        "SELECT * FROM episodes "
                        "WHERE state = 'active' AND session_id = ?"
                    )
                    params = [sid]
                    if since_iso:
                        query += " AND timestamp >= ?"
                        params.append(since_iso)
                    query += " ORDER BY timestamp DESC LIMIT ?"
                    params.append(int(max_per_session))
                    cursor.execute(query, tuple(params))
                    results.extend(dict(r) for r in cursor.fetchall())
                # Trim to max_total preserving newest-first ordering
                results.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
                results = results[: int(max_total)]
            return results
        finally:
            conn.close()

    def list_recent_session_ids(
        self,
        limit: int = 5,
        since_iso: Optional[str] = None,
    ) -> List[str]:
        """Phase 7: return the N most-recently-active session_ids."""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            query = "SELECT session_id FROM session_meta"
            params: list = []
            if since_iso:
                query += " WHERE last_active >= ?"
                params.append(since_iso)
            query += " ORDER BY last_active DESC LIMIT ?"
            params.append(int(limit))
            cursor.execute(query, tuple(params))
            return [r["session_id"] for r in cursor.fetchall() if r["session_id"]]
        finally:
            conn.close()

    def save_relation(self, relation) -> bool:
        """Persist a relation edge"""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO relations
                (id, subject_id, predicate, object_id, strength, created_at, bidirectional)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                relation.id,
                relation.subject_id,
                relation.predicate.value if hasattr(relation.predicate, 'value') else relation.predicate,
                relation.object_id,
                relation.strength,
                relation.created_at.isoformat(),
                1 if relation.bidirectional else 0,
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving relation: {e}")
            return False
        finally:
            conn.close()

    def get_recent_episodes(self, limit: int = 10) -> List[Dict]:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM episodes
                WHERE state = 'active'
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()

            return [
                {
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'concept_ids': json.loads(row['concept_ids']) if row['concept_ids'] else [],
                    'raw_content': row['raw_content'],
                    'context': json.loads(row['context']) if row['context'] else {},
                    'importance': (lambda imp: imp.get('overall', imp.get('novelty', 0)) if isinstance(imp, dict) else 0)(json.loads(row['importance_json'])) if row['importance_json'] else 0,
                    'source': row['source']
                }
                for row in rows
            ]
        finally:
            conn.close()

    def get_stats(self) -> Dict:
        conn = get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as count FROM concepts")
            concept_count = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM concepts WHERE state = 'suppressed'")
            suppressed_count = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM concepts WHERE state = 'archived'")
            archived_count = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM episodes WHERE state = 'active'")
            episode_count = cursor.fetchone()['count']

            return {
                'total_concepts': concept_count,
                'suppressed_count': suppressed_count,
                'archived_count': archived_count,
                'working_memory_size': episode_count
            }
        finally:
            conn.close()

    def get_all_concepts_raw(self, include_history: bool = True) -> List[Dict]:
        """Get all concepts from database"""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            if include_history:
                cursor.execute("SELECT * FROM concepts")
            else:
                cursor.execute("SELECT * FROM concepts WHERE state = 'active' AND COALESCE(is_current_version, 1) = 1")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_all_episodes_raw(self) -> List[Dict]:
        """Get all episodes from database"""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM episodes WHERE state = 'active' ORDER BY timestamp")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_all_relations_raw(self) -> List[Dict]:
        """Get all relation rows"""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM relations")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def clear_all(self):
        """Clear all data (for testing)"""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM concepts")
            cursor.execute("DELETE FROM episodes")
            cursor.execute("DELETE FROM relations")
            cursor.execute("DELETE FROM sleep_cycles")
            cursor.execute("DELETE FROM session_meta")
            conn.commit()
        finally:
            conn.close()

    def clear_concepts_relations(self):
        """Clear only semantic memory tables (concepts + relations)."""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM relations")
            cursor.execute("DELETE FROM concepts")
            conn.commit()
        finally:
            conn.close()

    def save_sleep_record(self, record: Dict) -> bool:
        """Persist a sleep cycle outcome so wake-summary sees it post-restart."""
        import uuid as _uuid
        conn = get_connection()
        try:
            cursor = conn.cursor()
            ts = record.get("timestamp") or utc_isoformat()
            cursor.execute(
                """
                INSERT OR REPLACE INTO sleep_cycles
                (id, start_time, end_time, nrem_duration, rem_duration,
                 memories_consolidated, memories_forgotten, dreams_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(_uuid.uuid4()),
                    ts,
                    ts,
                    float(record.get("duration", 0.0)) / 2.0,
                    float(record.get("duration", 0.0)) / 2.0,
                    int(record.get("consolidated", 0)),
                    int(record.get("forgotten", 0)),
                    json.dumps({
                        "mode": record.get("mode"),
                        "reason": record.get("reason"),
                        "dreams": record.get("dreams", 0),
                        "synced_concepts": record.get("synced_concepts", 0),
                        "synced_relations": record.get("synced_relations", 0),
                    }),
                ),
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving sleep record: {e}")
            return False
        finally:
            conn.close()

    def get_sleep_records_since(self, since_iso: Optional[str] = None) -> List[Dict]:
        """Return sleep cycle records since cutoff, newest first, in the engine
        sleep_history shape so WakeSummaryBuilder needs no changes."""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            if since_iso:
                cursor.execute(
                    "SELECT * FROM sleep_cycles WHERE start_time >= ? ORDER BY start_time DESC",
                    (since_iso,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM sleep_cycles ORDER BY start_time DESC LIMIT 200"
                )
            rows = cursor.fetchall()
            out: List[Dict] = []
            for row in rows:
                meta = {}
                try:
                    meta = json.loads(row["dreams_json"]) if row["dreams_json"] else {}
                except Exception:
                    meta = {}
                out.append({
                    "timestamp": row["start_time"],
                    "mode": meta.get("mode") or "deep",
                    "reason": meta.get("reason") or "forced",
                    "consolidated": int(row["memories_consolidated"] or 0),
                    "forgotten": int(row["memories_forgotten"] or 0),
                    "dreams": int(meta.get("dreams") or 0),
                    "duration": float((row["nrem_duration"] or 0) + (row["rem_duration"] or 0)),
                    "synced_concepts": int(meta.get("synced_concepts") or 0),
                    "synced_relations": int(meta.get("synced_relations") or 0),
                })
            return out
        finally:
            conn.close()

    def save_session_meta(self, session_id: str, message_count: int, total_concepts: int, total_sleeps: int):
        """Save session metadata"""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            now = utc_isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO session_meta
                (session_id, created_at, last_active, message_count, total_concepts, total_sleeps)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, now, now, message_count, total_concepts, total_sleeps))
            conn.commit()
        finally:
            conn.close()

    def load_session_meta(self, session_id: str) -> Optional[Dict]:
        """Load session metadata"""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM session_meta WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # ── Circadian sleep config (v0.7.7+) ─────────────────────────────────

    def get_user_sleep_config(self, user_id: str) -> Dict[str, Any]:
        """Return this user's sleep config. Returns the documented defaults
        when the user has no row yet — so the scheduler always gets a
        valid config without callers needing to insert first."""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM user_sleep_config WHERE user_id = ?", (user_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return {
                    "user_id": user_id,
                    "timezone": "UTC",
                    "sleep_start": "23:00",
                    "sleep_end": "07:00",
                    "enabled": True,
                    "last_sleep_at": None,
                    "is_default": True,
                }
            d = dict(row)
            d["enabled"] = bool(d.get("enabled", 1))
            d["is_default"] = False
            return d
        finally:
            conn.close()

    def save_user_sleep_config(
        self,
        user_id: str,
        timezone_name: Optional[str] = None,
        sleep_start: Optional[str] = None,
        sleep_end: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Upsert this user's sleep config. Only the fields the caller
        provided are updated; missing fields keep their existing or
        default value."""
        existing = self.get_user_sleep_config(user_id)
        new_tz = timezone_name if timezone_name is not None else existing["timezone"]
        new_start = sleep_start if sleep_start is not None else existing["sleep_start"]
        new_end = sleep_end if sleep_end is not None else existing["sleep_end"]
        new_enabled = (
            int(bool(enabled)) if enabled is not None
            else int(bool(existing.get("enabled", True)))
        )
        now = utc_isoformat()

        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_sleep_config
                  (user_id, timezone, sleep_start, sleep_end, enabled, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                  timezone   = excluded.timezone,
                  sleep_start = excluded.sleep_start,
                  sleep_end   = excluded.sleep_end,
                  enabled     = excluded.enabled,
                  updated_at  = excluded.updated_at
                """,
                (user_id, new_tz, new_start, new_end, new_enabled, now),
            )
            conn.commit()
        finally:
            conn.close()
        return self.get_user_sleep_config(user_id)

    def mark_user_slept(self, user_id: str, when_iso: Optional[str] = None) -> None:
        """Record that this user just completed a deep-sleep cycle. Used
        by the scheduler to enforce once-per-night."""
        when_iso = when_iso or utc_isoformat()
        conn = get_connection()
        try:
            cursor = conn.cursor()
            # Make sure a row exists first (insert with defaults if not).
            existing = self.get_user_sleep_config(user_id)
            if existing.get("is_default"):
                self.save_user_sleep_config(user_id)
            cursor.execute(
                "UPDATE user_sleep_config SET last_sleep_at = ? WHERE user_id = ?",
                (when_iso, user_id),
            )
            conn.commit()
        finally:
            conn.close()

    def list_user_sleep_configs(self) -> List[Dict[str, Any]]:
        """All users with explicit sleep configs. Used by the sweeper to
        iterate at scheduled-check time."""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_sleep_config")
            rows = cursor.fetchall()
            out = []
            for row in rows:
                d = dict(row)
                d["enabled"] = bool(d.get("enabled", 1))
                out.append(d)
            return out
        finally:
            conn.close()


# Initialize on import
_memory = None


def get_memory() -> SQLiteMemory:
    global _memory
    if _memory is None:
        _memory = SQLiteMemory()
    return _memory
