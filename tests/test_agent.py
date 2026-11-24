"""Unit and integration tests for SmartAgent tools, OpenAI function calling, memory, and retries."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from smartagent.agent import SmartAgent
from smartagent.config import Settings
from smartagent.tools import calculator, database_query


def test_calculator_safe_and_blocks_injection() -> None:
    res = calculator("(120 * 1.5) + sqrt(144)")
    assert res["ok"] is True
    assert res["result"] == 192

    bad = calculator("__import__('os').system('echo pwned')")
    assert bad["ok"] is False

    div0 = calculator("10 / 0")
    assert div0["ok"] is False


def test_database_query_read_only_and_seeded(tmp_path: Path) -> None:
    db_file = tmp_path / "test.db"
    res = database_query(
        "SELECT c.region, SUM(o.total_usd) AS rev FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.region",
        db_path=db_file,
    )
    assert res["ok"] is True
    assert res["row_count"] == 3

    blocked = database_query("DROP TABLE orders", db_path=db_file)
    assert blocked["ok"] is False
    assert "read-only" in blocked["error"].lower()


def test_openai_tool_calling_memory_and_retry() -> None:
    call_counter = {"n": 0}

    def fake_create(**kwargs):
        call_counter["n"] += 1
        # Simulate one transient API failure before succeeding.
        if call_counter["n"] == 1:
            raise ConnectionError("Simulated transient 503 error")
        if call_counter["n"] == 2:
            tc = SimpleNamespace(
                id="call_calc_1",
                function=SimpleNamespace(
                    name="calculator",
                    arguments=json.dumps({"expression": "250 * 4"}),
                ),
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tc]))]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="250 multiplied by 4 equals 1000.", tool_calls=[]
                    )
                )
            ]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    agent = SmartAgent(
        settings=Settings(openai_api_key="sk-test", max_retries=3),
        client=fake_client,
    )
    res = agent.ask("What is 250 * 4?")
    assert "1000" in res.answer
    assert res.mode_used == "openai"
    assert res.retries_used == 1
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].ok is True
    assert res.tool_calls[0].result["result"] == 1000
    assert len(agent.history) >= 4
