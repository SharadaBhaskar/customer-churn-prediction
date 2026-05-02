import sqlite3
import os
from werkzeug.security import generate_password_hash
from datetime import datetime

DATABASE = 'churnpredict.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row   # lets you access columns by name
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # ── 1. USERS (login, register, profile, settings pages) ────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name   TEXT    NOT NULL,
            email       TEXT    NOT NULL UNIQUE,
            password    TEXT    NOT NULL,
            phone       TEXT,
            role        TEXT    NOT NULL DEFAULT 'user',   -- 'admin' | 'user'
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # ── 2. CUSTOMERS (customers, add_customer, update_customer pages) ───────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name        TEXT    NOT NULL,
            email            TEXT    NOT NULL UNIQUE,
            phone            TEXT,
            age              INTEGER,
            plan_type        TEXT,        -- 'Select Plan' / 'Basic' / 'Premium' etc.
            tenure           INTEGER,     -- months
            monthly_charges  REAL,
            support_calls    INTEGER DEFAULT 0,
            contract_type    TEXT,        -- 'Month-to-Month' / 'One Year' / 'Two Year'
            payment_method   TEXT,
            status           TEXT    NOT NULL DEFAULT 'Active',  -- 'Active' | 'Churned'
            created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at       TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # ── 3. PREDICTIONS (prediction, prediction_result pages) ────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id   INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
            churn_prob    REAL    NOT NULL,   -- 0.0 – 1.0
            risk_level    TEXT    NOT NULL,   -- 'Low Risk' | 'Medium Risk' | 'High Risk'
            suggested_action TEXT,
            predicted_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # ── 4. SEGMENTS (segmentation page) ─────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS segments (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id    INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
            segment_label  TEXT    NOT NULL,  -- 'High Value' | 'At Risk' | 'Low Value' | 'New Customers'
            segment_score  REAL,
            segmented_at   TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # ── 5. RETENTION STRATEGIES (retention page) ────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retention_strategies (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            description TEXT,
            status      TEXT NOT NULL DEFAULT 'Active',  -- 'Active' | 'Inactive'
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # ── 6. SMS ALERTS (sms_alert page) ──────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sms_alerts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id   INTEGER REFERENCES customers(id) ON DELETE SET NULL,
            target_group  TEXT,   -- 'At Risk Customers' | 'All Customers' etc.
            message       TEXT NOT NULL,
            scheduled_at  TEXT,
            sent_at       TEXT,
            status        TEXT NOT NULL DEFAULT 'Pending'  -- 'Pending' | 'Sent' | 'Failed'
        )
    """)

    # ── 7. CAMPAIGNS (campaign, campaign_results pages) ─────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS campaigns (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT NOT NULL,
            target_segment TEXT,
            campaign_type  TEXT,   -- 'Email' | 'SMS' | 'Push' etc.
            start_date     TEXT,
            end_date       TEXT,
            status         TEXT NOT NULL DEFAULT 'Scheduled',  -- 'Scheduled' | 'Running' | 'Completed'
            created_at     TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS campaign_results (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id  INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
            sent         INTEGER DEFAULT 0,
            delivered    INTEGER DEFAULT 0,
            responses    INTEGER DEFAULT 0,
            conversions  INTEGER DEFAULT 0,
            recorded_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # ── 8. FEEDBACK (feedback page) ─────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER REFERENCES users(id) ON DELETE SET NULL,
            rating      INTEGER CHECK(rating BETWEEN 1 AND 5),
            message     TEXT,
            submitted_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # ── 9. SETTINGS (settings page) ─────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            key     TEXT    NOT NULL,   -- 'email_notifications' | 'sms_notifications' | 'theme' | 'language'
            value   TEXT    NOT NULL,
            UNIQUE(user_id, key)
        )
    """)

    # ── 10. ANALYTICS CACHE (analytics, dashboard pages) ────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analytics_cache (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_key   TEXT NOT NULL UNIQUE,  -- 'total_customers' | 'churn_rate' | 'avg_tenure' etc.
            metric_value TEXT NOT NULL,
            computed_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # ── 11. REPORTS (reports page) ──────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type   TEXT NOT NULL,   -- 'Churn Report' | 'Customer Report' | 'Revenue Report' | 'Campaign Report'
            generated_by  INTEGER REFERENCES users(id) ON DELETE SET NULL,
            filepath      TEXT,            -- path to downloaded file
            generated_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # ── SEED: default admin user ─────────────────────────────────────────────
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO users (full_name, email, password, role)
            VALUES (?, ?, ?, ?)
        """, (
            'Rahul Sharma',
            'rahul@mail.com',
            generate_password_hash('admin123'),
            'admin'
        ))

    # ── SEED: default retention strategies ──────────────────────────────────
    cursor.execute("SELECT COUNT(*) FROM retention_strategies")
    if cursor.fetchone()[0] == 0:
        strategies = [
            ('Discount Offer',       '10% discount for at-risk customers',          'Active'),
            ('Loyalty Program',      'Reward loyal customers with points',           'Active'),
            ('Personalized Support', 'Priority support for high-risk customers',     'Active'),
            ('Plan Upgrade Offer',   'Offer better plans to retain churning users',  'Inactive'),
        ]
        cursor.executemany(
            "INSERT INTO retention_strategies (title, description, status) VALUES (?, ?, ?)",
            strategies
        )

    conn.commit()
    conn.close()
    print("✅ Database initialized: churnpredict.db")


# ── Helper: quick query shortcuts used across routes ────────────────────────
def query_db(query, args=(), one=False):
    conn = get_db()
    cur = conn.execute(query, args)
    rv = cur.fetchall()
    conn.close()
    return (rv[0] if rv else None) if one else rv


def execute_db(query, args=()):
    conn = get_db()
    cur = conn.execute(query, args)
    conn.commit()
    last_id = cur.lastrowid
    conn.close()
    return last_id


if __name__ == '__main__':
    init_db()