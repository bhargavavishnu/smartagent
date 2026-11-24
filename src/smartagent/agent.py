"""SmartAgent core orchestrator: OpenAI tool-calling loop, memory, retries, and local router fallback."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from openai import OpenAI
from pydantic import BaseModel, Field

from smartagent.config import Settings
from smartagent.schemas import SYSTEM_PROMPT, TOOLS_SCHEMA
from smartagent.tools import calculator, database_query, web_search

_MATH_EXPR_RE = re.compile(
    r"(?:sqrt|log|log10|sin|cos|tan|abs|round|pow)\s*\([^)]+\)|[\d\.\(\)]+(?:\s*[\+\-\*\/\^\%]\s*[\d\.\(\)]+)+"
)
_DB_KEYWORDS = {
    "database", "sql", "sqlite", "order", "orders", "customer", "customers",
    "product", "products", "sku", "stock", "inventory", "revenue", "sales",
    "enterprise", "plan", "tier", "region", "delivered", "returned", "pending",
    "shipped", "unit_price", "price", "headphone", "keyboard", "webcam", "monitor", "hub",
}
_WEB_KEYWORDS = {
    "latest", "news", "weather", "current", "released", "release", "who won",
    "who is", "what is", "documentation", "docs", "python", "fastapi", "openai",
    "github", "langchain", "population", "capital", "ceo", "stock price", "search",
    "guidelines", "nist", "look",
}
_GREETING_OR_MEMORY_PHRASES = (
    "hello", "hi ", "hey", "who are you", "what can you do", "help",
    "earlier", "previous", "what did i ask", "remember", "summarize our chat",
)


class ToolTrace(BaseModel):
    """Execution record for a single tool invocation."""

    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    ok: bool
    attempts: int = 1


class AgentTurnResult(BaseModel):
    """Structured output returned by SmartAgent for one user turn."""

    answer: str
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    mode_used: str
    retries_used: int = 0
    latency_ms: float


def _with_retries(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay_s: float = 0.05,
) -> tuple[Any, int]:
    """Execute a callable with exponential backoff on transient exceptions."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn(), attempt
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(base_delay_s * (2**attempt))
    assert last_exc is not None
    raise last_exc


def _route_local_intent(
    user_message: str,
    history: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    """Deterministic structured router used when running locally without an OpenAI key."""
    text = user_message.strip()
    low = f" {text.lower()} "

    # 1. Explicit SQL query or obvious database analytics intent
    if re.match(r"^\s*(SELECT|WITH)\b", text, re.IGNORECASE):
        return "database_query", {"sql": text}

    # Conversational/memory check before keyword matching
    if any(phrase in low for phrase in _GREETING_OR_MEMORY_PHRASES) and not any(
        k in low for k in ("calculate", "sql", "search", "revenue", "orders", "stock")
    ):
        return None, {}

    # Check math expression or explicit arithmetic keywords
    math_match = _MATH_EXPR_RE.search(text.replace(",", ""))
    if math_match or any(w in low for w in ("calculate", "compute", "percent", "% of", "square root", "divide", "multiply")):
        if math_match:
            return "calculator", {"expression": math_match.group(0).strip()}
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", text.replace(",", ""), re.IGNORECASE)
        if pct_match:
            p, base = pct_match.groups()
            return "calculator", {"expression": f"({p} / 100) * {base}"}
        nums = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
        if len(nums) >= 2:
            if "divide" in low or "ratio" in low:
                return "calculator", {"expression": f"{nums[0]} / {nums[1]}"}
            if "multiply" in low or "times" in low:
                return "calculator", {"expression": f"{nums[0]} * {nums[1]}"}
            if "plus" in low or "add" in low or "sum" in low:
                return "calculator", {"expression": f"{nums[0]} + {nums[1]}"}
            if "minus" in low or "subtract" in low or "difference" in low:
                return "calculator", {"expression": f"{nums[0]} - {nums[1]}"}
        if "square root" in low and nums:
            return "calculator", {"expression": f"sqrt({nums[0]})"}

    # Check internal database intent
    tokens = set(re.findall(r"[a-z_]+", low))
    if tokens & _DB_KEYWORDS:
        sql = _nl_to_sqlite(low)
        return "database_query", {"sql": sql}

    # Check multi-turn follow-up referencing prior tool result (e.g., "double that", "add 10% to that")
    if ("that" in tokens or "it" in tokens) and history:
        for msg in reversed(history):
            if msg.get("role") == "tool":
                try:
                    payload = json.loads(msg["content"])
                    if "result" in payload:
                        val = payload["result"]
                        if "double" in low:
                            return "calculator", {"expression": f"{val} * 2"}
                        if "half" in low:
                            return "calculator", {"expression": f"{val} / 2"}
                except Exception:
                    pass

    # Web search fallback for external factual / current queries
    if tokens & _WEB_KEYWORDS or text.endswith("?"):
        return "web_search", {"query": text, "max_results": 4}

    return None, {}


def _nl_to_sqlite(low: str) -> str:
    """Translate common analytics questions into clean read-only SQLite queries."""
    if "revenue" in low and "region" in low:
        return (
            "SELECT c.region, ROUND(SUM(o.total_usd), 2) AS total_revenue "
            "FROM orders o JOIN customers c ON o.customer_id = c.id "
            "GROUP BY c.region ORDER BY total_revenue DESC"
        )
    if "total revenue" in low or "total sales" in low:
        return "SELECT ROUND(SUM(total_usd), 2) AS total_revenue, COUNT(*) AS order_count FROM orders"
    if "low stock" in low or "stock" in low or "inventory" in low:
        return "SELECT sku, name, category, stock_qty, unit_price FROM products ORDER BY stock_qty ASC"
    if "returned" in low or "pending" in low or "shipped" in low or "delivered" in low:
        for st in ("returned", "pending", "shipped", "delivered"):
            if st in low:
                return (
                    "SELECT o.id, c.name AS customer, p.name AS product, o.quantity, o.total_usd, o.order_date "
                    "FROM orders o JOIN customers c ON o.customer_id = c.id "
                    "JOIN products p ON o.product_id = p.id "
                    f"WHERE o.status = '{st}' ORDER BY o.order_date DESC"
                )
    if "enterprise" in low or "customer" in low or "tier" in low:
        if "enterprise" in low:
            return "SELECT id, name, region, plan_tier FROM customers WHERE plan_tier = 'Enterprise'"
        return "SELECT id, name, region, plan_tier FROM customers ORDER BY id"
    if "top" in low or "best" in low or "most" in low:
        return (
            "SELECT p.name, SUM(o.quantity) AS units_sold, ROUND(SUM(o.total_usd), 2) AS revenue "
            "FROM orders o JOIN products p ON o.product_id = p.id "
            "GROUP BY p.id ORDER BY revenue DESC"
        )
    if "order" in low:
        return (
            "SELECT o.id, c.name AS customer, p.name AS product, o.quantity, o.total_usd, o.status, o.order_date "
            "FROM orders o JOIN customers c ON o.customer_id = c.id "
            "JOIN products p ON o.product_id = p.id ORDER BY o.order_date DESC"
        )
    return "SELECT sku, name, category, unit_price, stock_qty FROM products ORDER BY unit_price DESC"


class SmartAgent:
    """Multi-tool assistant supporting OpenAI function calling, memory, and retry recovery."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        tool_registry: dict[str, Callable[..., dict[str, Any]]] | None = None,
        max_history_messages: int = 20,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.mode = self.settings.resolve_mode() if client is None else "openai"
        self.client = client or (
            OpenAI(api_key=self.settings.openai_api_key)
            if self.mode == "openai"
            else None
        )
        self.tools: dict[str, Callable[..., dict[str, Any]]] = tool_registry or {
            "calculator": calculator,
            "database_query": database_query,
            "web_search": web_search,
        }
        self.max_history_messages = max_history_messages
        self.history: list[dict[str, Any]] = []

    def reset_memory(self) -> None:
        """Clear conversational memory."""
        self.history.clear()

    def _trim_history(self) -> None:
        if len(self.history) > self.max_history_messages:
            self.history = self.history[-self.max_history_messages :]

    def _execute_tool(self, name: str, raw_args: dict[str, Any]) -> ToolTrace:
        if name not in self.tools:
            return ToolTrace(
                tool_name=name,
                arguments=raw_args,
                result={"ok": False, "error": f"Unknown tool '{name}'."},
                ok=False,
            )

        def _invoke() -> dict[str, Any]:
            res = self.tools[name](**raw_args)
            if not res.get("ok", False) and "failed" in str(res.get("error", "")).lower():
                # Retry transient network/search failures automatically.
                raise RuntimeError(res["error"])
            return res

        try:
            res, retries = _with_retries(_invoke, max_retries=self.settings.max_retries)
            return ToolTrace(
                tool_name=name,
                arguments=raw_args,
                result=res,
                ok=bool(res.get("ok", False)),
                attempts=retries + 1,
            )
        except Exception as exc:
            return ToolTrace(
                tool_name=name,
                arguments=raw_args,
                result={"ok": False, "error": str(exc)},
                ok=False,
                attempts=self.settings.max_retries,
            )

    def ask(self, user_message: str) -> AgentTurnResult:
        """Run one conversational turn with memory and dynamic tool routing."""
        if not user_message.strip():
            raise ValueError("User message must not be empty.")
        t0 = time.perf_counter()
        self.history.append({"role": "user", "content": user_message})

        if self.mode == "openai":
            result = self._run_openai(t0)
        else:
            result = self._run_local(user_message, t0)

        self._trim_history()
        return result

    def _run_openai(self, t0: float) -> AgentTurnResult:
        traces: list[ToolTrace] = []
        total_retries = 0
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history,
        ]

        for _ in range(self.settings.max_tool_rounds):
            response, retries = _with_retries(
                lambda: self.client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=0.0,
                ),
                max_retries=self.settings.max_retries,
            )
            total_retries += retries
            choice_msg = response.choices[0].message
            tool_calls = getattr(choice_msg, "tool_calls", None) or []

            if not tool_calls:
                answer = (choice_msg.content or "").strip()
                self.history.append({"role": "assistant", "content": answer})
                return AgentTurnResult(
                    answer=answer,
                    tool_calls=traces,
                    mode_used="openai",
                    retries_used=total_retries,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                )

            assistant_record = {
                "role": "assistant",
                "content": choice_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_record)
            self.history.append(assistant_record)

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    parsed_args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(parsed_args, dict):
                        raise ValueError("Tool arguments must decode to a JSON object.")
                    trace = self._execute_tool(fn_name, parsed_args)
                except Exception as exc:
                    trace = ToolTrace(
                        tool_name=fn_name,
                        arguments={"raw": tc.function.arguments},
                        result={"ok": False, "error": f"Invalid tool JSON arguments: {exc}"},
                        ok=False,
                    )

                total_retries += max(0, trace.attempts - 1)
                traces.append(trace)
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(trace.result),
                }
                messages.append(tool_msg)
                self.history.append(tool_msg)

        fallback_msg = "Reached maximum tool-calling rounds before completing the task."
        self.history.append({"role": "assistant", "content": fallback_msg})
        return AgentTurnResult(
            answer=fallback_msg,
            tool_calls=traces,
            mode_used="openai",
            retries_used=total_retries,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

    def _run_local(self, user_message: str, t0: float) -> AgentTurnResult:
        tool_name, args = _route_local_intent(user_message, self.history)
        traces: list[ToolTrace] = []
        retries_used = 0

        if tool_name is None:
            prior_user_msgs = [
                m["content"] for m in self.history[:-1] if m.get("role") == "user"
            ]
            if "earlier" in user_message.lower() or "previous" in user_message.lower() or "remember" in user_message.lower():
                if prior_user_msgs:
                    answer = f"Earlier in this conversation you asked: '{prior_user_msgs[-1]}'."
                else:
                    answer = "This is the first message in our current conversation."
            else:
                answer = (
                    "Hello! I am SmartAgent. I can evaluate calculations (`calculator`), "
                    "query the company analytics database (`database_query`), or search the web (`web_search`)."
                )
        else:
            trace = self._execute_tool(tool_name, args)
            traces.append(trace)
            retries_used = max(0, trace.attempts - 1)
            self.history.append(
                {
                    "role": "tool",
                    "tool_call_id": f"local_{len(self.history)}",
                    "content": json.dumps(trace.result),
                }
            )
            answer = _format_local_answer(tool_name, trace.result)

        self.history.append({"role": "assistant", "content": answer})
        return AgentTurnResult(
            answer=answer,
            tool_calls=traces,
            mode_used="local",
            retries_used=retries_used,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )


def _format_local_answer(tool_name: str, result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"Tool `{tool_name}` reported an error: {result.get('error', 'Unknown error')}"
    if tool_name == "calculator":
        return f"`{result['expression']}` = **{result['result']}**"
    if tool_name == "database_query":
        rows = result.get("rows", [])
        if not rows:
            return "The query executed cleanly and returned 0 rows."
        preview = "\n".join(str(r) for r in rows[:8])
        return f"Database query returned {result['row_count']} row(s):\n{preview}"
    if tool_name == "web_search":
        hits = result.get("results", [])
        if not hits:
            return "No web results matched your query."
        lines = [f"- **{h['title']}** ({h['url']}): {h['snippet']}" for h in hits[:3]]
        return "Top web search findings:\n" + "\n".join(lines)
    return json.dumps(result)
