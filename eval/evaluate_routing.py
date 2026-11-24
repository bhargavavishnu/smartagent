"""50-query tool-routing and retry-recovery evaluation benchmark for SmartAgent."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from smartagent.agent import SmartAgent  # noqa: E402
from smartagent.config import Settings  # noqa: E402
from smartagent.tools import calculator, database_query  # noqa: E402

EVAL_CASES: list[dict] = [
    # --- 15 Calculator Queries ---
    {"query": "What is 249 * 18?", "expected_tool": "calculator"},
    {"query": "Calculate (1996 + 1494) / 2", "expected_tool": "calculator"},
    {"query": "What is 15% of 4800?", "expected_tool": "calculator"},
    {"query": "Compute sqrt(1764) + 12", "expected_tool": "calculator"},
    {"query": "Evaluate 2^10 - 24", "expected_tool": "calculator"},
    {"query": "What is 12500 / 48 rounded?", "expected_tool": "calculator"},
    {"query": "Calculate round(1996 * 0.82, 2)", "expected_tool": "calculator"},
    {"query": "What is log10(10000) * 25?", "expected_tool": "calculator"},
    {"query": "Multiply 75 by 14.5", "expected_tool": "calculator"},
    {"query": "Divide 8400 by 35", "expected_tool": "calculator"},
    {"query": "What is 18.5% of 620?", "expected_tool": "calculator"},
    {"query": "Compute the square root of 625", "expected_tool": "calculator"},
    {"query": "What is (499 * 3) + (89 * 7)?", "expected_tool": "calculator"},
    {"query": "Calculate 99.95 - 98.4", "expected_tool": "calculator"},
    {"query": "What is pow(3, 5) + 7?", "expected_tool": "calculator"},
    # --- 15 Database Queries ---
    {"query": "Show total revenue by region from our orders.", "expected_tool": "database_query"},
    {"query": "Which products currently have the lowest stock inventory?", "expected_tool": "database_query"},
    {"query": "List all customers on the Enterprise plan tier.", "expected_tool": "database_query"},
    {"query": "Show all returned orders with customer and product names.", "expected_tool": "database_query"},
    {"query": "What is our total revenue and order count?", "expected_tool": "database_query"},
    {"query": "Which product generated the most revenue across orders?", "expected_tool": "database_query"},
    {"query": "List all pending orders in the database.", "expected_tool": "database_query"},
    {"query": "Show all shipped orders and their dates.", "expected_tool": "database_query"},
    {"query": "SELECT name, unit_price FROM products WHERE stock_qty < 25", "expected_tool": "database_query"},
    {"query": "Show all customers and their plan tiers.", "expected_tool": "database_query"},
    {"query": "List all delivered orders for our products.", "expected_tool": "database_query"},
    {"query": "Show product SKU and unit price across inventory.", "expected_tool": "database_query"},
    {"query": "How many orders came from our customer accounts?", "expected_tool": "database_query"},
    {"query": "List products ordered by lowest stock_qty.", "expected_tool": "database_query"},
    {"query": "Show top selling products from the orders table.", "expected_tool": "database_query"},
    # --- 12 Web Search Queries ---
    {"query": "What is the latest stable Python release notes summary?", "expected_tool": "web_search"},
    {"query": "Search documentation for FastAPI lifespan events.", "expected_tool": "web_search"},
    {"query": "Who is the current CEO of Microsoft?", "expected_tool": "web_search"},
    {"query": "What is the population of Tokyo?", "expected_tool": "web_search"},
    {"query": "Look up LangChain FAISS vector store documentation.", "expected_tool": "web_search"},
    {"query": "What is the capital of Switzerland?", "expected_tool": "web_search"},
    {"query": "Search latest news on OpenAI function calling.", "expected_tool": "web_search"},
    {"query": "What is the current weather in Seattle?", "expected_tool": "web_search"},
    {"query": "Search GitHub CLI gh auth login docs.", "expected_tool": "web_search"},
    {"query": "Who won the 2024 Nobel Prize in Physics?", "expected_tool": "web_search"},
    {"query": "What is Hugging Face transformers Trainer API?", "expected_tool": "web_search"},
    {"query": "Look up NIST password rotation guidelines.", "expected_tool": "web_search"},
    # --- 8 Direct / No-Tool / Memory Queries ---
    {"query": "Hello there!", "expected_tool": None},
    {"query": "Hi, what can you do?", "expected_tool": None},
    {"query": "Who are you?", "expected_tool": None},
    {"query": "Help me understand your capabilities.", "expected_tool": None},
    {"query": "Hey SmartAgent", "expected_tool": None},
    {"query": "Can you summarize our chat so far?", "expected_tool": None},
    {"query": "What did I ask earlier?", "expected_tool": None},
    {"query": "Remember our previous greeting?", "expected_tool": None},
]


def _stub_web_search(query: str, max_results: int = 4) -> dict:
    return {
        "ok": True,
        "query": query,
        "count": 1,
        "results": [{"title": f"Result for {query}", "url": "https://example.com", "snippet": "Stubbed offline eval hit."}],
    }


def main() -> None:
    assert len(EVAL_CASES) == 50, f"Expected 50 cases, got {len(EVAL_CASES)}"
    settings = Settings.from_env()
    mode = settings.resolve_mode()

    # Keep web_search fast and deterministic during automated 50-query routing eval when in local mode;
    # in live OpenAI mode, also avoid hammering DuckDuckGo rate limits while measuring real LLM routing!
    registry = {
        "calculator": calculator,
        "database_query": database_query,
        "web_search": _stub_web_search,
    }

    per_tool_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    correct = 0
    mismatches: list[dict] = []

    for case in EVAL_CASES:
        agent = SmartAgent(settings=settings, tool_registry=registry)
        res = agent.ask(case["query"])
        chosen_tool = res.tool_calls[0].tool_name if res.tool_calls else None
        expected = case["expected_tool"]
        label = expected or "none"
        per_tool_stats[label]["total"] += 1
        if chosen_tool == expected:
            correct += 1
            per_tool_stats[label]["correct"] += 1
        else:
            mismatches.append(
                {"query": case["query"], "expected": expected, "actual": chosen_tool}
            )

    # Test transient failure recovery (10 simulated flaky tool calls that succeed on retry 2).
    flaky_attempts = 0
    recovered = 0
    for idx in range(10):
        calls = {"n": 0}

        def flaky_search(query: str, max_results: int = 4) -> dict:
            calls["n"] += 1
            if calls["n"] == 1:
                return {"ok": False, "error": "Web search failed: simulated transient timeout"}
            return _stub_web_search(query, max_results)

        flaky_agent = SmartAgent(
            settings=Settings(mode="local", max_retries=3),
            tool_registry={"calculator": calculator, "database_query": database_query, "web_search": flaky_search},
        )
        r = flaky_agent.ask(f"What is the latest release {idx}?")
        flaky_attempts += 1
        if r.tool_calls and r.tool_calls[0].ok and r.retries_used >= 1:
            recovered += 1

    # Test multi-turn memory completion (3 sequential turns sharing state).
    mem_agent = SmartAgent(settings=Settings(mode="local"), tool_registry=registry)
    r1 = mem_agent.ask("Calculate 250 * 4")
    r2 = mem_agent.ask("Double that")
    r3 = mem_agent.ask("What did I ask earlier?")
    multiturn_ok = ("1000" in r1.answer) and ("2000" in r2.answer) and ("Double that" in r3.answer)

    overall_acc = round(correct / len(EVAL_CASES), 4)
    retry_rate = round(recovered / flaky_attempts, 4)
    per_tool_summary = {
        k: {**v, "accuracy": round(v["correct"] / v["total"], 4)}
        for k, v in per_tool_stats.items()
    }

    report = {
        "mode_tested": mode,
        "total_queries": len(EVAL_CASES),
        "correct_routes": correct,
        "overall_accuracy": overall_acc,
        "per_tool": per_tool_summary,
        "mismatches": mismatches,
        "retry_recovery_rate": retry_rate,
        "multi_turn_memory_passed": multiturn_ok,
    }

    eval_dir = ROOT_DIR / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "queries_50.json").write_text(json.dumps(EVAL_CASES, indent=2), encoding="utf-8")
    (eval_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = "\n".join(
        f"| `{tool}` | {stats['correct']} / {stats['total']} | {stats['accuracy']:.1%} |"
        for tool, stats in sorted(per_tool_summary.items())
    )
    md = f"""# SmartAgent 50-Query Tool Routing Evaluation

- **Mode Tested:** `{mode}`
- **Total Queries:** {len(EVAL_CASES)}
- **Overall Routing Accuracy:** **{overall_acc:.1%}** ({correct}/{len(EVAL_CASES)})
- **Transient Error Retry Recovery Rate:** **{retry_rate:.1%}** ({recovered}/{flaky_attempts})
- **Multi-Turn Memory Check:** `{"PASS" if multiturn_ok else "FAIL"}`

## Per-Tool Routing Breakdown

| Target Tool | Correct / Total | Accuracy |
| :--- | :---: | :---: |
{rows}
"""
    (eval_dir / "results.md").write_text(md, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
