"""
Router Langchain Fix — final, tested.
Changes versus the failing version:
• Uses ChatOpenAI.bind_tools so the model really returns structured tool calls.
• Extracts arguments from resp.tool_calls[0] via .args / ["args"], falling back to the old OpenAI schema when needed – so KeyError is gone.
• Keeps fuzzy column suggestions and language‑detection logic.
• Demo test‑suite at the bottom.
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
    """Parse markdown‑style schema description into {table: [columns…]}."""
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
Return JSON with keys: route, reason, suggestions, follow_up.
Respond in language: {lang}.
Hints: {json.dumps(hints)}
Known columns (truncated): {'; '.join(_FLAT_COLUMNS[:MAX_SCHEMA_LINES_IN_PROMPT])}
""".strip()

# ───────── OpenAI function schema ─────────
function_schema = {
    "name": "route_decision",
    "description": "Decide if query needs SQL generation or clarification.",
    "parameters": {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ["sql_query", "clarify"]},
            "reason": {"type": "string"},
            "suggestions": {"type": "array", "items": {"type": "string"}},
            "follow_up": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["route", "reason", "suggestions"],
    },
}

# ChatOpenAI instance with bound tools
llm = ChatOpenAI(model=MODEL, api_key=API_KEY).bind_tools([function_schema])

# ───────── Main helper ─────────

def decide_route(question: str) -> dict:
    """Return routing decision JSON for a single *question*."""
    hints = fuzzy_suggest(question)
    lang = detect(question)

    messages = [
        {"role": "system", "content": build_system_prompt(hints, lang)},
        {"role": "user", "content": question},
    ]

    resp = llm.invoke(messages)

    # LangChain always provides .tool_calls – ensure it's there
    if not getattr(resp, "tool_calls", None):
        raise RuntimeError(
            "🛑 Model replied without tool_call. Check prompt or lower temperature."
        )

    tc = resp.tool_calls[0]

    # --- Extract arguments in a version‑safe way ------------------
    if isinstance(tc, dict):
        data = tc.get("args", {})
        # Fallback for older raw‑OpenAI style
        if not data and "function" in tc:
            data = json.loads(tc["function"]["arguments"])
    else:  # ToolCall dataclass
        data = tc.args

    # --- Validate & patch suggestions -----------------------------
    valid_suggestions = [s for s in data.get("suggestions", []) if s in _FLAT_COLUMNS]

    if data.get("route") == "clarify":
        valid_suggestions = []
    elif len(valid_suggestions) < 3:
        extra = [s for s in hints if s not in valid_suggestions]
        valid_suggestions.extend(extra[: 3 - len(valid_suggestions)])

    data["suggestions"] = valid_suggestions
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
        print("Route      :", res.get("route"))
        print("Reason     :", res.get("reason"))
        print("Suggestions:", ", ".join(res.get("suggestions", [])) or "None")
        print("Follow‑up  :", ", ".join(res.get("follow_up", [])) or "None")
        print("Language   :", res.get("language"))
