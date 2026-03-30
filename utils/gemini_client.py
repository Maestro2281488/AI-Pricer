"""
utils/gemini_client.py

Покращення:
- Збільшено таймаут до 12 секунд, щоб Gemini встигав аналізувати великі тексти та JSON.
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
_TIMEOUT_SEC = 12.0   # ЗБІЛЬШЕНО! 4.5 сек було замало для безкоштовного API


def _build_model():
    return genai.GenerativeModel(
        model_name=_MODEL_NAME,
        generation_config={
            "temperature":        0.15,
            "max_output_tokens":  2048,
            "response_mime_type": "application/json",
        },
        system_instruction=(
            "You are a JSON-only pricing API. "
            "Never output any text outside the JSON object. "
            "Be concise — explanation max 2 sentences, advice max 2 sentences."
        ),
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
                print(f"🔍 Gemini raw ({len(text)} chars): {text[:400]}")
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
        if debug and result:
            print(f"🔍 Gemini async raw ({len(result)} chars): {result[:400]}")
        return result
    except asyncio.TimeoutError:
        print(f"⏱ Gemini timeout ({timeout}s) — використовуємо ML fallback")
        return None


# ── Допоміжні ─────────────────────────────────────────────────────────────────

def _build_contents(prompt: str, images=None) -> list:
    contents = [prompt]
    if images:
        photo_instruction = (
            "\n\n=== ФОТО ТОВАРУ ===\n"
            "Проаналізуй надані зображення і визнач:\n"
            "1. Видимий стан (подряпини, потертості, дефекти)\n"
            "2. Комплектність (коробка, аксесуари)\n"
            "3. Відповідність опису\n"
            "Врахуй це у своїй оцінці ціни.\n"
        )
        contents[0] = prompt + photo_instruction
        contents.extend(images)
    return contents