"""
utils/gemini_client.py
Gemini API клієнт з:
- generation_config для чистого JSON
- retry логікою
- debug режимом
"""

import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

_MODEL_NAME = "gemini-2.5-flash"
_MAX_RETRIES = 2


def get_gemini_response(
    prompt: str,
    images=None,
    model_name: str = _MODEL_NAME,
    debug: bool = False,
) -> str | None:
    """
    Відправляє prompt у Gemini, повертає текст відповіді.

    Parameters
    ----------
    prompt     : текст промпту
    images     : список Pillow-зображень або None
    model_name : назва моделі
    debug      : якщо True — друкує перші 400 символів відповіді
    """
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature":        0.15,   # менше творчості = стабільніший JSON
            "max_output_tokens":  8192,
            "response_mime_type": "application/json",  # примусовий JSON-mode
        },
        system_instruction=(
            "You are a JSON-only pricing API. "
            "Never output any text outside the JSON object."
        ),
    )

    contents = [prompt]
    if images:
        contents.extend(images)

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
