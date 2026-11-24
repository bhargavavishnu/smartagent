"""Web search tool using DuckDuckGo (ddgs) with retry-ready clean error handling."""

from __future__ import annotations

from typing import Any
from ddgs import DDGS


def web_search(query: str, max_results: int = 4) -> dict[str, Any]:
    """Search the public web for up-to-date information via DuckDuckGo."""
    cleaned = query.strip()
    if not cleaned:
        return {"ok": False, "error": "Search query must not be empty."}
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(cleaned, max_results=max(1, min(max_results, 8))))
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw_results
        ]
        return {"ok": True, "query": cleaned, "count": len(results), "results": results}
    except Exception as exc:
        return {"ok": False, "query": cleaned, "error": f"Web search failed: {exc}"}
