"""OpenAI tool JSON schemas and structured routing system prompt for SmartAgent."""

from __future__ import annotations

SYSTEM_PROMPT = """You are SmartAgent, a precise multi-tool AI assistant with conversational memory.

Available tools and routing rules:
1. `calculator`: Use for ALL numerical computations, percentages, compound growth, ratios, conversions, or algebraic expressions. Never calculate complex arithmetic mentally.
2. `database_query`: Use for questions about internal company products, SKUs, stock inventory, customers, regions, plan tiers, orders, revenue, or order status.
   Database schema (SQLite, read-only):
   - products(id, sku, name, category, unit_price, stock_qty)
   - customers(id, name, region, plan_tier)
   - orders(id, customer_id, product_id, quantity, total_usd, status, order_date)
3. `web_search`: Use for external world knowledge, current events, software documentation, public facts, or online lookups not present in the internal database.
4. Direct response (no tool): Use for greetings, clarifying questions, summarizing prior turns in conversation memory, or capability explanations.

When a tool returns an error, inspect the error message and fix your arguments (for example, adjusting SQL column names or math syntax) before retrying."""

TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate mathematical expressions accurately (arithmetic, percentages, "
                "exponents, sqrt, log, trig). Input must be a valid math expression."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '(1494 * 1.15) - 200' or 'sqrt(144) * 3.5'.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "database_query",
            "description": (
                "Run a read-only SQLite SELECT query against the company analytics database "
                "with tables: products(id, sku, name, category, unit_price, stock_qty), "
                "customers(id, name, region, plan_tier), and "
                "orders(id, customer_id, product_id, quantity, total_usd, status, order_date)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single read-only SELECT or WITH SQLite query.",
                    }
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the public web for external facts, live news, documentation, "
                "or information outside the internal company database."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concise search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-8).",
                        "default": 4,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]
