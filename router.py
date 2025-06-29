from __future__ import annotations
import os, re, json, pathlib
from typing import List, Dict
from rapidfuzz import process, fuzz  # type: ignore
from openai import OpenAI  # type: ignore
import langdetect

# ─────── configuration ───────
MODEL = "gpt-4o-mini"
DB_DESC_FILE = pathlib.Path("instructions/db_description.txt")
MAX_SCHEMA_LINES_IN_PROMPT = 120
SIMILARITY_THRESHOLD = 80
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# ─────── 1.  Parse db_description.txt ───────
# ─────── 1.  Parse db_description.txt ───────
def _parse_db_description(path: pathlib.Path) -> Dict[str, List[str]]:
    schema: Dict[str, List[str]] = {}
    current_table = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        tbl = re.match(r"^###\s+([A-Za-z0-9_]+)", raw)
        if tbl:
            current_table = tbl.group(1).lower()
            schema[current_table] = []
            continue

        if current_table:
            # NEW: прибираємо маркери ' - ' на початку
            line = raw.lstrip(" \t-")
            col = re.match(r"^([A-Za-z0-9_]+):", line)
            if col:
                schema[current_table].append(col.group(1).lower())

    return schema

_SCHEMA = _parse_db_description(DB_DESC_FILE)
_FLAT_COLUMNS = [f"{tbl}.{col}" for tbl, cols in _SCHEMA.items() for col in cols]

# ─────── 2.  Fast local “did‑you‑mean” search ───────
def _fuzzy_suggest(text: str, k: int = 3) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]{3,}", text.lower())

    # 🔄 NEW: накопичуємо найкращий скор для кожного cand
    best: dict[str, int] = {}
    for tok in tokens:
        for cand, score, _ in process.extract(
            tok,
            _FLAT_COLUMNS,
            scorer=fuzz.partial_ratio,   # partial працює ліпше для «chocolate» vs products.name
            limit=20,                    # ширше, ніж 3
        ):
            if score >= 65:              # ↓ трохи нижчий поріг
                best[cand] = max(best.get(cand, 0), score)

    # відсортуємо й віддамо top-k
    return sorted(best, key=best.get, reverse=True)[:k]



# ─────── 3.  Prompts and function spec ───────
SYSTEM_PROMPT = f"""
You are a router for a SQL chat assistant.

You must *always* return a JSON object with the keys:
  "route"        – "sql_query" or "clarify"
  "reason"       – short explanation (match user language)
  "suggestions"  – array of up to 3 table.column strings
  "follow_up"    – array with 0–2 clarifying questions (match user language)

Mandatory Rules
===============
• Choose **"sql_query"** when the question can be answered *using any column from the known tables*.  
  ─ If the query mentions a word that fuzzy-matches ≥ 1 column (hint supplied below), treat it as data-related.  
  ─ For “top N” / aggregate / date-filtered questions, proceed with SQL generation.
• Choose **"clarify"** only when **none** of the known columns seem relevant **or** the question is clearly outside business/database context.
• Never invent table or column names – use only those in the *Known tables* section.
• If the user gives a vague time period (“this month”, “last year”), assume the current year **or** ask for a concrete date range.

Clarification Guidance
======================
• If the query names a **person, product, or store** but lacks an ID/slug, ask for a specific identifier (e.g. customer_id, product_id, store_id).  
• If the query refers to a **time period** but does not specify concrete dates, ask the user to provide start/end dates.  
• If multiple interpretations are possible, ask the *most critical* 1-2 follow-up questions to disambiguate.
• If you choose "clarify", leave "suggestions" empty.

Contextual Awareness
====================
• **customers** → `customers` table (customer-level info)  
• **orders / order items** → `orders`, `order_items` (sales & quantities)  
• **products & categories** → `products`  
• **campaigns & budgets** → `campaigns`  
• **inventory per store** → `inventory`, joined with `stores`  
• **regions & stores** → `regions`, `stores`  
(Use joins across these tables as needed.)

Known tables/columns (truncated for context):
{'; '.join(_FLAT_COLUMNS[:MAX_SCHEMA_LINES_IN_PROMPT])}
""".strip()


FUNCTION_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "decide_action",
            "parameters": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "string",
                        "enum": ["sql_query", "clarify"],
                    },
                    "reason": {"type": "string"},
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "follow_up": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["route", "reason", "suggestions", "follow_up"],
            },
        },
    }
]

# ─────── 4.  Public helper ───────
def decide_route(question: str, *, history: List[dict] | None = None) -> dict:
    hints = _fuzzy_suggest(question)
    detected_lang = langdetect.detect(question)

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "system", "content": f"Suggestions hint: {json.dumps(hints)}"},
        {"role": "user", "content": question},
    ]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        tools=FUNCTION_SPEC,
        tool_choice="required",
    )

    choice = completion.choices[0].message

    if choice.tool_calls and choice.tool_calls[0].function.arguments:
        response = json.loads(choice.tool_calls[0].function.arguments)

        original_suggestions = response.get("suggestions", [])
        valid_suggestions = [s for s in original_suggestions if s in _FLAT_COLUMNS]

        if len(valid_suggestions) < 3:
            additional = [s for s in hints if s in _FLAT_COLUMNS and s not in valid_suggestions]
            valid_suggestions += additional[: 3 - len(valid_suggestions)]

        if response.get("route") == "clarify":
            valid_suggestions = [] 

        response["suggestions"] = valid_suggestions
        response["language"] = detected_lang
        return response

    return {"error": "Model did not produce a function call"}

def test_schema() -> None:
    """Переконуємось, що парсер прочитав схему БД без спотворень."""
    expected_cols: dict[str, set[str]] = {
        # ─── MERCHANDISING ────────────────────────────────────────────
        "customers": {"id", "full_name", "email", "city", "created_at"},
        "products": {"id", "name", "category", "price_cents", "created_at"},
        "orders": {"id", "customer_id", "total_price_cents", "created_at"},
        "order_items": {
            "id", "order_id", "product_id", "quantity", "price_cents", "created_at"
        },

        # ─── MARKETPLACE ──────────────────────────────────────────────
        "regions": {"id", "country", "state", "city", "created_at"},
        "stores": {"id", "name", "region_id", "active", "created_at"},
        "employees": {
            "id", "full_name", "store_id", "position", "hire_date", "created_at"
        },

        # ─── ANALYTICS ────────────────────────────────────────────────
        "campaigns": {
            "id", "name", "status", "budget_cents",
            "start_date", "end_date", "created_at"
        },
        "inventory": {"id", "product_id", "store_id", "quantity", "created_at"},
    }

    # перетворюємо парс-результат у множини для коректного порівняння
    parsed_cols = {tbl: set(cols) for tbl, cols in _SCHEMA.items()}

    assert parsed_cols == expected_cols, (
        "❌ Parsed schema does not match expected!\n\n"
        f"Parsed  : {parsed_cols}\n"
        f"Expected: {expected_cols}"
    )




def run_tests():
    test_cases = [
        # English
        "List all products containing peanuts",
        "How many customers bought chocolate flavor?",
        "Why are sky blue?",
        "How calculate average value?",

        # Ukrainian translations (4)
        "Покажи кампанії з бюджетом більше ніж 50000",
        "Який продукт є найпопулярнішим цього місяця?",
        "Скільки замовлень зробив Джон Сміт?",
        "Покажи користувачів із підтвердженим номером телефону",

        # Polish translations (4)
        "Pokaż transakcje dla sklepu 'best-store-123'",
        "Jaki jest wynik NPS sprzedawcy 'John Doe'?",
        "Pokaż wszystkie kampanie z budżetem powyżej 50 tysięcy",
        "Dlaczego niebo jest niebieskie?",
    ]

    total_invalid = 0
    for idx, question in enumerate(test_cases, 1):
        print(f"\n{'='*30}\nTest {idx}: {question}\n{'-'*30}")
        result = decide_route(question)
        route = result.get("route")
        reason = result.get("reason")
        suggestions = result.get("suggestions", [])
        follow_up = result.get("follow_up", [])
        lang = result.get("language", "en")

        print(f"Language    : {lang}")
        print(f"Route       : {route}")
        print(f"Reason      : {reason}")
        print(f"Suggestions : {', '.join(suggestions) or 'None'}")
        print(f"Follow-up   : {', '.join(follow_up) or 'None'}")

        invalid = [s for s in suggestions if s not in _FLAT_COLUMNS]
        if invalid:
            total_invalid += 1
            print(f"  Warning: Invalid suggestions detected: {', '.join(invalid)}")

    print(f"\n\u2705 Finished {len(test_cases)} tests. Invalid suggestions found in {total_invalid} cases.")

if __name__ == "__main__":
    run_tests()

from pprint import pprint
pprint(_SCHEMA)          # має містити всі таблиці й колонки
print(len(_FLAT_COLUMNS), "columns total")

# у будь-якому REPL або у кінці скрипта
from pprint import pprint

pprint(_SCHEMA)          # словник {table: [columns…]}
print(len(_FLAT_COLUMNS), "columns total")
