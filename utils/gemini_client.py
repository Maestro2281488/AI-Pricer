"""
utils/gemini_client.py

Покращення:
- Впроваджено Strict JSON Schema для гарантованої структури відповіді.
- Додано жорсткий детермінізм (temperature=0.0, top_k=1, top_p=0.1).
- Оновлено SYSTEM_PROMPT з акцентом на компаративи.
"""

import os
import time
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

_MODEL_NAME  = "gemini-2.5-flash-lite"
_MAX_RETRIES = 2
_TIMEOUT_SEC = 12.0

def _build_model():
    # Визначаємо жорстку схему JSON (Structured Outputs)
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "recommended_price": {"type": "INTEGER"},
            "price_range": {
                "type": "OBJECT",
                "properties": {
                    "min": {"type": "INTEGER"},
                    "max": {"type": "INTEGER"}
                },
                "required": ["min", "max"]
            },
            "strategies": {
                "type": "OBJECT",
                "properties": {
                    "FAST": {"type": "INTEGER"},
                    "BALANCED": {"type": "INTEGER"},
                    "MAX_PROFIT": {"type": "INTEGER"}
                },
                "required": ["FAST", "BALANCED", "MAX_PROFIT"]
            },
            "explanation": {"type": "STRING"},
            "key_factors": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "advice": {"type": "STRING"}
        },
        "required": [
            "recommended_price", "price_range", "strategies",
            "explanation", "key_factors", "advice"
        ]
    }

    # Твій покращений промпт англійською
    system_instruction = (
        "You are a professional AI pricer for the 'monobazar' platform within the monobank app. "
        "Your goal is to help sellers set a fair market price for used goods. "
        "You receive a product description, ML model forecast, and real examples of sold items (comparables). "
        "\n\nSTRICT RULES:\n"
        "1. Always be objective and rely on comparables data for standard items.\n"
        "2. Formulate 3 strategies: FAST, BALANCED, MAX_PROFIT.\n"
        "3. CRITICAL: If the item is marked as RARE, COLLECTIBLE, ANTIQUE, or a SET, do NOT anchor to the low median of standard items. Value its uniqueness appropriately and trust high ML forecasts.\n"
        "4. Your response must be EXCLUSIVELY in a valid JSON format. No markdown, no extra text.\n"
        "5. Respond ONLY in Ukrainian."
    )

    return genai.GenerativeModel(
        model_name=_MODEL_NAME,
        generation_config={
            "temperature": 0.0,  # Мінімум випадковості
            "top_p": 0.1,
            "top_k": 1,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
            "response_schema": response_schema, # Підключення схеми
        },
        system_instruction=system_instruction,
    )

# ── Синхронний виклик ─────────────────────────────────────────────────────────

def get_gemini_response(
    prompt: str,
    images=None,
    debug: bool = False,
) -> str | None:
    model    = _build_model()
    contents = _build_contents(prompt, images)

    for attempt in range(1, _MAX_RETRIES + 2):
        try:
            response = model.generate_content(contents)
            text = response.text.strip()
            if debug:
                print(f"🔍 Gemini raw: {text[:400]}")
            return text
        except Exception as exc:
            print(f"❌ Gemini error attempt {attempt}: {exc}")
            if attempt <= _MAX_RETRIES:
                time.sleep(2 ** attempt)
    return None

# ── Асинхронний виклик з таймаутом ───────────────────────────────────────────

async def get_gemini_response_async(
    prompt: str,
    images=None,
    debug: bool = False,
    timeout: float = _TIMEOUT_SEC,
) -> str | None:
    loop  = asyncio.get_event_loop()
    model = _build_model()
    contents = _build_contents(prompt, images)

    def _call():
        try:
            return model.generate_content(contents).text.strip()
        except Exception as exc:
            print(f"❌ Gemini async error: {exc}")
            return None

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _call),
            timeout=timeout,
        )
        return result
    except asyncio.TimeoutError:
        print(f"⏱ Gemini timeout ({timeout}s) — використовую ML fallback")
        return None

# ── Допоміжні ─────────────────────────────────────────────────────────────────

def _build_contents(prompt: str, images=None) -> list:
    contents = [prompt]
    if images:
        photo_instruction = (
            "\n\n=== PRODUCT PHOTOS ===\n"
            "Analyze these images to determine:\n"
            "1. Visual condition (scratches, wear, defects).\n"
            "2. Completeness (box, accessories).\n"
            "3. Consistency with the description.\n"
        )
        contents[0] = prompt + photo_instruction
        contents.extend(images)
    return contents