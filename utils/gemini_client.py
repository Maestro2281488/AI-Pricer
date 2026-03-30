"""
utils/gemini_client.py

Покращення:
- async get_gemini_response_async() для паралельного запуску з ML
- Таймаут 4 сек → якщо не встигло, ML fallback (критерій №2: < 5 сек)
- Стрімінг як опція
- Photo analysis prompt якщо є зображення
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
_TIMEOUT_SEC = 4.5   # якщо Gemini не відповів за 4.5 сек — ML fallback


def _build_model():
    return genai.GenerativeModel(
        model_name=_MODEL_NAME,
        generation_config={
            "temperature":        0.15,
            "max_output_tokens":  2048,   # зменшено для швидкості
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
    """Синхронний виклик з retry."""
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
    """
    Асинхронний виклик Gemini з таймаутом.
    Якщо не встигло за timeout секунд → повертає None (ML fallback).
    """
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


# ── Стрімінг (опційно, для streaming endpoint) ───────────────────────────────

def get_gemini_streaming(prompt: str, images=None):
    """
    Генератор що стрімить токени від Gemini.
    Використовуй у Flask SSE або FastAPI StreamingResponse.

    Приклад:
        for chunk in get_gemini_streaming(prompt):
            print(chunk, end="", flush=True)
    """
    model    = _build_model()
    contents = _build_contents(prompt, images)
    try:
        for chunk in model.generate_content(contents, stream=True):
            if chunk.text:
                yield chunk.text
    except Exception as exc:
        print(f"❌ Gemini streaming error: {exc}")
        return


# ── Допоміжні ─────────────────────────────────────────────────────────────────

def _build_contents(prompt: str, images=None) -> list:
    """
    Якщо є зображення — додаємо до контенту і розширюємо промпт.
    Gemini аналізує стан товару за фото.
    """
    contents = [prompt]
    if images:
        # Додаємо інструкцію для аналізу фото перед зображеннями
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