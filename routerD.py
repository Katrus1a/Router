"""router_prompt_refactor.py — full script with HTML chat‑style test output

Run locally → it will generate `router_chat_tests.html` next to the script,
which opens in any browser and shows user/assistant bubbles.
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
            tok, _FLAT_COLUMNS, scorer=fuzz.partial_ratio, limit=20
        ):
            if score >= 65:
                best[cand] = max(best.get(cand, 0), score)

    return sorted(best, key=best.get, reverse=True)[:k]

# ───────── Prompt builder ─────────

def build_system_prompt(hints: List[str], lang: str) -> str:
    """
    Build SYSTEM prompt enforcing one‑paragraph `message` and proper routing.
    (Kept identical to previous canvas version.)
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

# Low temperature but not zero (allows a little creativity)
llm = ChatOpenAI(model=MODEL, api_key=API_KEY, temperature=0.05).bind_tools([function_schema])

# ───────── Main helper ─────────

def decide_route(question: str) -> dict:
    """Return routing decision JSON for a single *question*."""
    hints = fuzzy_suggest(question)
    lang = detect(question)

    messages = [
        {"role": "system", "content": build_system_prompt(hints, lang)},
        {"role": "user", "content": question},
    ]

    resp = llm.invoke(messages, tool_choice="required")
    if not getattr(resp, "tool_calls", None):
        raise RuntimeError("🛑 Model replied without tool_call. Check prompt or lower temperature.")

    tc = resp.tool_calls[0]
    if isinstance(tc, dict):
        data = tc.get("args") or tc.get("arguments")
        if not data and "function" in tc:
            data = json.loads(tc["function"]["arguments"])
    else:
        data = tc.args

    data["language"] = lang
    return data

# ───────── Pretty HTML output ─────────

CSS = """
body{font-family:Arial,Helvetica,sans-serif;background:#000000;margin:0;padding:20px;}
.chat{max-width:800px;margin:auto;}
.bubble{padding:10px 14px;border-radius:16px;margin:8px 0;display:inline-block;max-width:75%;}
.user{background:#5ea2ef;align-self:flex-start;}
.assistant{background:#d6d6f5;align-self:flex-end;}
.row{display:flex;flex-direction:column;}
.idx{color:#aeb5ff;font-size:12px;margin-top:4px;}
"""


def save_chat_html(results: List[dict], path: pathlib.Path) -> None:
    """Generate a messenger‑style HTML file from results list."""
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Router Chat Tests</title>",
        f"<style>{CSS}</style></head><body><div class='chat'>",
    ]
    for r in results:
        parts.append("<div class='row'>")
        parts.append(f"<div class='bubble user'><strong>User:</strong> {r['query']}</div>")
        parts.append(
            f"<div class='bubble assistant'><strong>Assistant ({r['route']}):</strong> {r['message']}</div>"
        )
        parts.append(f"<div class='idx'>Test {r['idx']} • lang={r['language']}</div>")
        parts.append("</div>")
    parts.append("</div></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")

# ───────── Demo tests ─────────

def run_tests() -> None:
    tests = [
        # Products & Orders
        "List all gluten-free products",
        "Which three products had the highest average order quantity last month?",
        "Скільки клієнтів купили товар 'Galaxy Earbuds' щонайменше двічі?",
        "Show me orders that used discount codes",
        # Campaigns
        "Compare ROAS for Holiday Sale 2024 vs 2025",
        "List active campaigns ending within the next 10 days",
        "Pokaż kampanie z kosztami powyżej 100 000 zł",
        # Employees & Stores
        "Who is the longest-tenured store manager?",
        "Який середній стаж роботи співробітників у кожному регіоні?",
        "List employees with overdue performance reviews",
        "Czy mamy w sklepach więcej kasjerów czy magazynierów?",
        # Inventory & Regions
        "Which stores have fewer than 5 units left of any Electronics product?",
        "Total stock value by region (in dollars)",
        "Покажи регіони без жодного активного магазину",
        # KPI & Misc
        "Has average order value increased year-over-year?",
        "List customers who haven’t ordered since 2023",
        "How many first-time buyers did we have this quarter?",
        "Jaki jest współczynnik retencji klientów po 90 dniach?",
        "Чи є група Бумбокс популярною в Україні"
    ]

    results = []
    for i, q in enumerate(tests, 1):
        decision = decide_route(q)
        results.append({
            "idx": i,
            "query": q,
            "route": decision["route"],
            "message": decision["message"],
            "language": decision["language"],
        })

    html_path = pathlib.Path(__file__).with_name("router_chat_tests.html")
    save_chat_html(results, html_path)
    print(f"✅ Chat‑style report saved to: {html_path}")
    # Optionally open in default browser
    try:
        import webbrowser
        webbrowser.open(html_path.as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    run_tests()
