"""Interactive Streamlit UI for SmartAgent with tool execution traces and 50-query eval view."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from smartagent.agent import SmartAgent
from smartagent.config import Settings

ROOT_DIR = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT_DIR / "eval" / "results.json"

st.set_page_config(page_title="SmartAgent — Multi-Tool LLM Assistant", page_icon="🤖", layout="wide")
st.title("SmartAgent — Multi-Tool Assistant with Function Calling")
st.caption("Dynamic routing across calculator, DuckDuckGo web search, and read-only SQLite analytics DB.")

with st.sidebar:
    st.header("Agent Settings")
    api_key = st.text_input("OpenAI API Key (optional)", type="password")
    if st.button("Reset Conversation Memory", use_container_width=True):
        st.session_state.pop("agent", None)
        st.session_state.chat_log = []

    settings = Settings.from_env(openai_api_key_override=api_key or None)
    st.info(f"Active Mode: **{settings.resolve_mode().upper()}**")

if "agent" not in st.session_state or st.session_state.get("last_key") != api_key:
    st.session_state.agent = SmartAgent(settings=settings)
    st.session_state.last_key = api_key
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

tab_chat, tab_eval = st.tabs(["Interactive Agent", "50-Query Routing Benchmark"])

with tab_chat:
    for entry in st.session_state.chat_log:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("traces"):
                with st.expander("Tool Call Trace"):
                    st.json(entry["traces"])

    prompt = st.chat_input("Try: 'What is revenue by region?' or 'Calculate sqrt(144) * 18.5'")
    if prompt:
        st.session_state.chat_log.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            res = st.session_state.agent.ask(prompt)
            st.markdown(res.answer)
            st.caption(f"Mode: `{res.mode_used}` | Retries: `{res.retries_used}` | Latency: `{res.latency_ms} ms`")
            traces = [t.model_dump() for t in res.tool_calls]
            if traces:
                with st.expander("Tool Call Trace"):
                    st.json(traces)
            st.session_state.chat_log.append(
                {"role": "assistant", "content": res.answer, "traces": traces}
            )

with tab_eval:
    if EVAL_PATH.exists():
        data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Routing Accuracy", f"{data['overall_accuracy']:.1%}")
        c2.metric("Queries Evaluated", data["total_queries"])
        c3.metric("Retry Recovery Rate", f"{data['retry_recovery_rate']:.1%}")
        st.json(data)
    else:
        st.info("Run `uv run python eval/evaluate_routing.py` to generate evaluation metrics.")
