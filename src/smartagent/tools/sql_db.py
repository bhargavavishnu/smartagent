"""Read-only SQLite database query tool with seeded retail/SaaS analytics tables."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "company_analytics.db"

_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


def ensure_sample_db(db_path: Path = DEFAULT_DB_PATH) -> Path:
    """Create and seed a realistic sample SQLite database if it does not already exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and db_path.stat().st_size > 0:
        return db_path

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                sku TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                unit_price REAL NOT NULL,
                stock_qty INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                region TEXT NOT NULL,
                plan_tier TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                product_id INTEGER NOT NULL REFERENCES products(id),
                quantity INTEGER NOT NULL,
                total_usd REAL NOT NULL,
                status TEXT NOT NULL,
                order_date TEXT NOT NULL
            );

            INSERT INTO products (id, sku, name, category, unit_price, stock_qty) VALUES
                (1, 'ANC-PRO-100', 'AeroSound ANC Headphones', 'Audio', 249.00, 42),
                (2, 'KEY-MECH-75', 'TactilePro 75% Keyboard', 'Peripherals', 159.00, 18),
                (3, 'CAM-4K-HDR', 'ClearStream 4K Webcam', 'Video', 129.00, 65),
                (4, 'HUB-USB4-PRO', 'DockMaster USB4 Hub', 'Accessories', 89.00, 9),
                (5, 'MON-27-UHD', 'Lumina 27-inch 4K Monitor', 'Displays', 499.00, 23);

            INSERT INTO customers (id, name, region, plan_tier) VALUES
                (1, 'Northstar Labs', 'North America', 'Enterprise'),
                (2, 'BrightWave Media', 'Europe', 'Pro'),
                (3, 'Kestrel Robotics', 'North America', 'Enterprise'),
                (4, 'Solstice Design', 'APAC', 'Starter'),
                (5, 'Vertex Systems', 'Europe', 'Pro');

            INSERT INTO orders (id, customer_id, product_id, quantity, total_usd, status, order_date) VALUES
                (101, 1, 5, 4, 1996.00, 'delivered', '2026-10-02'),
                (102, 2, 1, 6, 1494.00, 'delivered', '2026-10-08'),
                (103, 3, 2, 10, 1590.00, 'shipped', '2026-10-14'),
                (104, 4, 3, 3, 387.00, 'delivered', '2026-10-19'),
                (105, 1, 4, 8, 712.00, 'delivered', '2026-10-25'),
                (106, 5, 1, 5, 1245.00, 'pending', '2026-11-03'),
                (107, 3, 5, 3, 1497.00, 'delivered', '2026-11-11'),
                (108, 2, 4, 7, 623.00, 'returned', '2026-11-15');
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def database_query(sql: str, db_path: Path = DEFAULT_DB_PATH, max_rows: int = 25) -> dict[str, Any]:
    """Execute a strictly read-only SELECT/WITH SQL query against the analytics SQLite DB."""
    ensure_sample_db(db_path)
    cleaned = sql.strip().rstrip(";")
    if not cleaned:
        return {"ok": False, "error": "SQL query must not be empty."}
    if ";" in cleaned:
        return {"ok": False, "error": "Multiple SQL statements are not allowed."}
    if _FORBIDDEN_SQL.search(cleaned):
        return {"ok": False, "error": "Only read-only SELECT queries are permitted."}
    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return {"ok": False, "error": "Query must begin with SELECT or WITH."}

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(cleaned)
            rows = [dict(row) for row in cur.fetchmany(max_rows)]
            columns = [desc[0] for desc in (cur.description or [])]
        finally:
            conn.close()
        return {
            "ok": True,
            "sql": cleaned,
            "columns": columns,
            "row_count": len(rows),
            "rows": rows,
        }
    except Exception as exc:
        return {"ok": False, "sql": cleaned, "error": str(exc)}
