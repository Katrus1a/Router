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
    """
    Produce the SYSTEM prompt that drives tool‑calling LLMs in *router* mode.

    The LLM must answer **only** JSON of the form
        {"route": "sql_query" | "clarify", "message": "<string>"}

    ░░░ message rules ░░░  (ONE paragraph, no section labels)
    • 1‑3 sentences, separated by a single space.
    • If the user explicitly asks to *list, show, count, find, average, filter* —
      i.e. retrieve rows — choose `route = "sql_query"`, even if columns or
      tables are not provided.
    • For `sql_query`:
        – Summarise the query intent.
        – Mention key columns naturally (e.g. “using columns orders.id and
          orders.total_price_cents”) **without square brackets** if the *hints*
          list is non‑empty; otherwise omit columns.
        – Optionally add one clarifying question.
    • For `clarify`:
        – State that more information is required.
        – Integrate at least **one** follow‑up question into the paragraph.
        – If the request is about *evaluating, comparing effectiveness, or
          optimisation* but provides no concrete metrics/columns, prefer
          `route = "clarify"` and ask which metrics to use.
    • Do not repeat the user’s wording verbatim.
    • Always respond in the user’s language (detected = {lang}).

    Examples
    --------
    sql_query →
    {"route":"sql_query","message":"Listing all products that contain peanuts using columns products.name and products.ingredients. Do you also need price information?"}

    clarify →
    {"route":"clarify","message":"I need more details to create a query. Which table or columns should I use?"}

    Any deviation from this format will be rejected by the calling code.
    """

    # build short context for the model
    known = '; '.join(_FLAT_COLUMNS[:MAX_SCHEMA_LINES_IN_PROMPT])
    hints_json = json.dumps(hints)

    return f"""
You are a router for a SQL chat assistant. Respond with **only** valid JSON:
{{"route": "sql_query" | "clarify", "message": "<string>"}}

RULES FOR `message` (one paragraph):
• 1–3 sentences, single paragraph.
• If the user asks to list/show/count/find/average/filter data, set route = "sql_query".
• sql_query → mention key columns naturally if any: {hints_json or '[]'}; no square brackets.
• clarify  → include at least one question; use it also for requests about performance/effectiveness without metrics.
• No repeats of user wording.
• Reply in the same language as the user ({lang}).

If you break the format, the response will be rejected.

Context (do NOT mention verbatim):
  • today = {today}
  • known columns (truncated): {known}
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
        "List the five most expensive Electronics products",
        "Покажи загальну виручку (у доларах) за замовленнями, зробленими 15 січня 2025 року",
        "Jaka będzie pogoda w Krakowie jutro?",
        "Has our customer engagement improved this quarter?",
        "Наскільки ефективна кампанія Back to School порівняно з минулим роком?",
        "Хто з співробітників показав найкращий результат і потребує підвищення?"
    ]

    for i, q in enumerate(tests, 1):
        print("\n" + "=" * 30)
        print(f"Test {i}: {q}")
        res = decide_route(q)
        print("Route   :", res.get("route"))
        print("Message :", res.get("message"))
        print("Language:", res.get("language"))
