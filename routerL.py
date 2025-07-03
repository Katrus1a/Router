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
      i.e. retrieve rows — choose `route = "sql_query"`, **but only if** the
      requested attributes exist in the known schema.
    • If the request mentions attributes that are **not present** in the schema
      (e.g. “gluten‑free”, “discount codes”, “costs”, “performance reviews”),
      choose `route = "clarify"` and ask the user to rephrase or map to an
      available column.
    • For `sql_query`:
        – Summarise the query intent.
        – **Name at least two fully‑qualified columns (schema.table.column) if hints are available**, e.g. “using columns orders.id and orders.total_price_cents”, *without square brackets*.
        – If the answer could meaningfully change with a date range, **append a concise follow‑up question asking for the desired period**.
        – Optionally add one more clarifying question (limit, sorting, etc.).
    • For `clarify`:
        – State that more information is required or that the field is missing.
        – Integrate at least **one** follow‑up question into the paragraph.
        – Also use `clarify` for requests about *evaluating, comparing
          effectiveness, optimisation* without explicit metrics/columns.
    • **You MUST write the `message` in the same language as the user**. If
      `lang = 'uk'`, reply in Ukrainian; if `lang = 'pl'`, reply in Polish, etc.
    • Do not repeat the user’s wording verbatim.

    Examples (multilingual)
    -----------------------
    Missing field → (EN)
    {"route":"clarify","message":"I can’t find a column related to discount codes in the database. Could you specify another attribute or table?"}

    sql_query → (UK)
    {"route":"sql_query","message":"Перелічую всі продукти, що містять арахіс, використовуючи колонки products.name і products.ingredients. Потрібно також вивести ціну?"}

    clarify → (PL)
    {"route":"clarify","message":"Potrzebuję więcej szczegółów, aby stworzyć zapytanie. Które tabele lub kolumny mam użyć?"}

    Any deviation from this format will be rejected by the calling code.
    """

    known = '; '.join(_FLAT_COLUMNS[:MAX_SCHEMA_LINES_IN_PROMPT])
    hints_json = json.dumps(hints)

    return f"""
You are a router for a SQL chat assistant. Respond with **only** valid JSON:
{{"route": "sql_query" | "clarify", "message": "<string>"}}

RULES FOR `message` (one paragraph):
• 1–3 sentences, single paragraph.
• If the user asks to list/show/count/find/average/filter data *and* all requested attributes exist, set route = "sql_query".
• If the request references attributes not present in the schema (e.g. “gluten‑free”, “discount codes”, “costs”, “performance reviews”), set route = "clarify".
• sql_query → mention **at least two fully‑qualified columns** if any: {hints_json or '[]'}; no square brackets. If the result depends on a time period, ask a brief follow‑up question for the desired date range.
• clarify  → include at least one question; also use when performance/effectiveness is asked without metrics.
• **Write the message in the same language as the user** (detected = {lang}).
• No repeats of user wording.

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

llm = ChatOpenAI(model=MODEL, api_key=API_KEY, temperature=0.01).bind_tools([function_schema])

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
        # ------------- Products & Orders -------------
        "List all gluten-free products",                                  # clarify
        "Which three products had the highest average order quantity last month?",  # sql_query
        "Скільки клієнтів купили товар 'Galaxy Earbuds' щонайменше двічі?",         # sql_query
        "Show me orders that used discount codes",                       # clarify

        # --------------- Campaigns -------------------
        "Compare ROAS for Holiday Sale 2024 vs 2025",                    # clarify
        "List active campaigns ending within the next 10 days",          # sql_query
        "Pokaż kampanie z kosztami powyżej 100 000 zł",                  # clarify

        # ----------- Employees & Stores --------------
        "Who is the longest-tenured store manager?",                     # sql_query
        "Який середній стаж роботи співробітників у кожному регіоні?",   # sql_query
        "List employees with overdue performance reviews",               # clarify
        "Czy mamy w sklepach więcej kasjerów czy magazynierów?",         # clarify

        # ----------- Inventory & Regions -------------
        "Which stores have fewer than 5 units left of any Electronics product?",    # sql_query
        "Total stock value by region (in dollars)",                      # clarify
        "Покажи регіони без жодного активного магазину",                 # sql_query

        # ----- Cross-schema KPIs & Misc --------------
        "Has average order value increased year-over-year?",             # sql_query
        "List customers who haven’t ordered since 2023",                 # sql_query
        "How many first-time buyers did we have this quarter?",          # sql_query
        "Jaki jest współczynnik retencji klientów po 90 dniach?"         # clarify
    ]

    for i, q in enumerate(tests, 1):
        print("\n" + "=" * 30)
        print(f"Test {i}: {q}")
        res = decide_route(q)
        print("Route   :", res.get("route"))
        print("Message :", res.get("message"))
        print("Language:", res.get("language"))
