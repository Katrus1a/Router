"""
Router Langchain — unified response version (Reason + Suggestions + Follow‑up → one user‑friendly text).

Fix 2025‑07‑01
==============
• Removed duplicated typo in `llm = …` line that caused `SyntaxError: unmatched ')'`.
• No functional changes otherwise; temperature‑0 and forced `tool_choice` remain.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from datetime import datetime
from typing import Dict, List

from langdetect import detect
from langchain_openai import ChatOpenAI
from rapidfuzz import fuzz, process

# ───────────── CONFIG ─────────────
MODEL = "gpt-4o-mini"
DB_DESC_FILE = pathlib.Path("instructions/db_description.txt")
MAX_SCHEMA_LINES_IN_PROMPT = 120
API_KEY = os.getenv("OPENAI_API_KEY")

today = datetime.today().strftime("%Y-%m-%d")

# ───────── db_description.txt → schema ─────────

def _parse_db_description(path: pathlib.Path) -> Dict[str, List[str]]:
    """Parse markdown-style schema description into {table: [columns…]}."""
    schema: Dict[str, List[str]] = {}
    current: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        tbl = re.match(r"^###\s+([A-Za-z0-9_]+)", line)
        if tbl:
            current = tbl.group(1).lower()
            schema[current] = []
            continue

        if current:
            col = re.match(r"^[ \t\-]*([A-Za-z0-9_]+):", line)
            if col:
                schema[current].append(col.group(1).lower())

    return schema


_SCHEMA = _parse_db_description(DB_DESC_FILE)
_FLAT_COLUMNS = [f"{t}.{c}" for t, cols in _SCHEMA.items() for c in cols]

# ───────── Local fuzzy suggestions ─────────

def fuzzy_suggest(text: str, k: int = 3) -> List[str]:
    """Return up to *k* column names similar to tokens in *text*."""
    tokens = re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
    best: dict[str, int] = {}

    for tok in tokens:
        for cand, score, _ in process.extract(
            tok,
            _FLAT_COLUMNS,
            scorer=fuzz.partial_ratio,
            limit=20,
        ):
            if score >= 65:
                best[cand] = max(best.get(cand, 0), score)

    return sorted(best, key=best.get, reverse=True)[:k]

# ───────── Prompt builder ─────────

def build_system_prompt(hints: List[str], lang: str) -> str:
    """Compose concise system instructions for the router model."""
    return f"""
You are a router for a SQL chat assistant.

Return **only** a JSON object with keys:
  • "route"   – "sql_query" or "clarify"
  • "message" – one coherent, user‑friendly string in **{lang}** that combines:
      – a short reason for the chosen route,
      – up to 3 useful column hints (if relevant),
      – 1‑2 follow‑up questions if clarification is needed.

Write the message in the same language as the user's question. Be polite, concise, and helpful.

Context you can use (do **not** mention it explicitly):
  • Today is {today}
  • Column hints: {json.dumps(hints)}
  • Known columns (truncated): {'; '.join(_FLAT_COLUMNS[:MAX_SCHEMA_LINES_IN_PROMPT])}
""".strip()

# ───────── OpenAI function schema ─────────
function_schema = {
    "name": "route_decision",
    "description": "Decide routing and craft a user‑friendly explanatory message.",
    "parameters": {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ["sql_query", "clarify"]},
            "message": {"type": "string"},
        },
        "required": ["route", "message"],
    },
}

# ChatOpenAI instance with bound tools (temperature 0 for determinism)
llm = ChatOpenAI(model=MODEL, api_key=API_KEY, temperature=0).bind_tools([function_schema])

# ───────── Main helper ─────────

def decide_route(question: str) -> dict:
    """Return routing decision JSON for a single *question*."""
    hints = fuzzy_suggest(question)
    lang = detect(question)

    messages = [
        {"role": "system", "content": build_system_prompt(hints, lang)},
        {"role": "user", "content": question},
    ]

    # Force the model to choose the `route_decision` tool every time
        # LangChain/OpenAI accept only "none", "auto", or "required" here;
    # since we bound **one** tool, "required" guarantees it will be used.
    resp = llm.invoke(messages, tool_choice="required")

    if not getattr(resp, "tool_calls", None):
        raise RuntimeError("🛑 Model replied without tool_call. Check prompt or lower temperature.")

    tc = resp.tool_calls[0]

    # Extract arguments (LangChain format first)
    if isinstance(tc, dict):
        data = tc.get("args", {}) or tc.get("arguments", {})
        if not data and "function" in tc:  # raw OpenAI fallback
            data = json.loads(tc["function"]["arguments"])
    else:  # ToolCall dataclass
        data = tc.args

    data["language"] = lang
    return data


# ───────── Demo tests ─────────
if __name__ == "__main__":
    tests = [
        "List all products containing peanuts",
        "How many customers bought chocolate flavor?",
        "Why are sky blue?",
        "Покажи користувачів із підтвердженим номером телефону",
        "Скільки замовлень зробив Джон Сміт?",
        "Pokaż wszystkie kampanie z budżetem powyżej 50 tysięcy",
    ]

    for i, q in enumerate(tests, 1):
        print("\n" + "=" * 30)
        print(f"Test {i}: {q}")
        res = decide_route(q)
        print("Route   :", res.get("route"))
        print("Message :", res.get("message"))
        print("Language:", res.get("language"))
