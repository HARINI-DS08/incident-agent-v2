"""SQLite persistence. Every state change is committed before the HTTP response is sent."""
import json
import os
import sqlite3
import threading

DB_PATH = os.environ.get("DB_PATH", "incidents.sqlite3")
_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_conn.execute("PRAGMA journal_mode=WAL")
_conn.executescript("""
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, body_hash TEXT NOT NULL, secrets TEXT NOT NULL,
  state TEXT NOT NULL, first_response TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS receipts (
  run_id TEXT NOT NULL, receipt_id TEXT NOT NULL, body_hash TEXT NOT NULL, response TEXT NOT NULL,
  PRIMARY KEY (run_id, receipt_id));
CREATE TABLE IF NOT EXISTS plan_cache (content_hash TEXT PRIMARY KEY, plan TEXT NOT NULL);
""")


def get_run(run_id):
    with _lock:
        row = _conn.execute("SELECT body_hash, secrets, state, first_response FROM runs WHERE run_id=?",
                            (run_id,)).fetchone()
    if not row:
        return None
    return {"body_hash": row[0], "secrets": json.loads(row[1]), "state": json.loads(row[2]),
            "first_response": row[3]}


def create_run(run_id, body_hash, secrets, state, first_response):
    with _lock:
        _conn.execute("INSERT INTO runs VALUES (?,?,?,?,?)",
                      (run_id, body_hash, json.dumps(secrets), json.dumps(state), first_response))


def get_receipt(run_id, receipt_id):
    with _lock:
        row = _conn.execute("SELECT body_hash, response FROM receipts WHERE run_id=? AND receipt_id=?",
                            (run_id, receipt_id)).fetchone()
    return {"body_hash": row[0], "response": row[1]} if row else None


def save_receipt(run_id, receipt_id, body_hash, state, response):
    """Store the receipt and the new run state in one transaction."""
    with _lock:
        _conn.execute("BEGIN")
        try:
            _conn.execute("INSERT INTO receipts VALUES (?,?,?,?)", (run_id, receipt_id, body_hash, response))
            _conn.execute("UPDATE runs SET state=? WHERE run_id=?", (json.dumps(state), run_id))
            _conn.execute("COMMIT")
        except Exception:
            _conn.execute("ROLLBACK")
            raise


def cached_plan(content_hash):
    with _lock:
        row = _conn.execute("SELECT plan FROM plan_cache WHERE content_hash=?", (content_hash,)).fetchone()
    return json.loads(row[0]) if row else None


def cache_plan(content_hash, plan):
    with _lock:
        _conn.execute("INSERT OR REPLACE INTO plan_cache VALUES (?,?)", (content_hash, json.dumps(plan)))
