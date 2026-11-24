"""Tool registry package for SmartAgent."""

from smartagent.tools.calculator import calculator
from smartagent.tools.sql_db import database_query, ensure_sample_db
from smartagent.tools.web_search import web_search

__all__ = ["calculator", "database_query", "ensure_sample_db", "web_search"]
