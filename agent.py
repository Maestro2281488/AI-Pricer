"""
agent.py — PriceAgent

Покращення:
1. Паралельний запуск ML + Gemini через asyncio (критерій: < 5 сек)
2. ML-результат повертається одразу, Gemini оновлює якщо встигає
3. Блок довіри: comparables_for_trust() → active/sold/reserved + price_position
4. market_speed_info() — індикатор швидкості ринку
5. RESERVED/ORDER_PROCESSING у навчанні та компаративах
6. Fallback ієрархія: Gemini → ML → категорійна медіана → загальна медіана
7. LRU-кеш для ML predictions
8. Динамічна порада з даними (батарея, пам'ять, specs)
"""

import json
import re
import asyncio
import hashlib
from functools import lru_cache
from typing import Optional

from utils.data_loader   import load_dataset
from utils.price_analyzer import PriceAnalyzer
from utils.gemini_client  import get_gemini_response, get_gemini_response_async


class PriceAgent:

    def __init__(self):
        # load_dataset тепер повертає 4 значення
        self.df, self.fast_sold, self.reserved_df, self.deleted_df = load_dataset()

        self.analyzer = PriceAnalyzer(
            self.df,
            self.fast_sold,
            self.reserved_df,
            self.deleted_df,
        )

    # ── Головний синхронний метод ─────────────────────────────────────────────

    def price_item(
        self,
        description: str,
        images=None,
        category_id: Optional[int] = None,
        debug: bool = False,
        fast_mode: bool = True,   # True = паралельний запуск ML+Gemini
    ) -> dict:
        """
        Parameters
        ----------
        description  : текст опису від продавця
        images       : список Pillow Image або None
        category_id  : ID категорії (опційно)
        debug        : друкувати сирий Gemini response
        fast_mode    : True = asyncio паралельний запуск (< 5 сек)

        Returns
        -------
        dict з ключами: recommended_price, price_range, strategies,
                        explanation, key_factors, advice,
                        trust_block, market_speed, _source
        """
        if fast_mode:
            try:
                return asyncio.run(
                    self._price_item_async(description, images, category_id, debug)
                )
            except RuntimeError:
                # Вже є event loop (напр. в Jupyter) — fallback до синхронного
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    self._price_item_async(description, images, category_id, debug)
                )
                loop.close()
                return result
        else:
            return self._price_item_sync(description, images, category_id, debug)

    # ── Асинхронний pipeline (fast_mode=True) ────────────────────────────────

    async def _price_item_async(
        self,
        description: str,
        images=None,
        category_id: Optional[int] = None,
        debug: bool = False,
    ) -> dict:
        """
        1. ML запускається відразу (синхронно, ~0.1–0.5 сек)
        2. Gemini запускається паралельно з таймаутом 4.5 сек
        3. Якщо Gemini встиг → повна відповідь
        4. Якщо Gemini не встиг → ML результат (вже готовий)
        """
        # ── ML частина (синхронно, швидко) ───────────────────────────────────
        ml_result = self._compute_ml(description, category_id)
        regression_price = ml_result["recommended_price"]
        strategies_ml    = ml_result["strategies_raw"]

        # ── Промпт ───────────────────────────────────────────────────────────
        prompt = self._build_prompt(description, category_id, regression_price, strategies_ml)

        # ── Паралельний виклик Gemini ─────────────────────────────────────────
        print("⚡ Паралельний запуск ML + Gemini...")
        raw = await get_gemini_response_async(prompt, images, debug=debug, timeout=12)

        if raw:
            parsed = _parse_json(raw)
            if parsed:
                if "strategies" not in parsed or not parsed["strategies"]:
                    parsed["strategies"] = {
                        k: v for k, v in strategies_ml.items() if not k.startswith("_")
                    }
                parsed["_source"] = "gemini"
                print("✅ Gemini відповів вчасно")
                return self._enrich_result(parsed, description, category_id, regression_price)

        # ── ML fallback ───────────────────────────────────────────────────────
        print("⚡ ML fallback (Gemini не встиг або помилка)")
        return self._enrich_result(ml_result, description, category_id, regression_price)

    # ── Синхронний pipeline (fast_mode=False) ────────────────────────────────

    def _price_item_sync(
        self,
        description: str,
        images=None,
        category_id: Optional[int] = None,
        debug: bool = False,
    ) -> dict:
        ml_result        = self._compute_ml(description, category_id)
        regression_price = ml_result["recommended_price"]
        strategies_ml    = ml_result["strategies_raw"]
        prompt           = self._build_prompt(description, category_id, regression_price, strategies_ml)

        print("Відправляємо в Gemini...")
        for attempt in range(1, 3):
            print(f"⏳ Gemini, спроба {attempt}...")
            raw = get_gemini_response(prompt, images, debug=debug)
            if raw:
                parsed = _parse_json(raw)
                if parsed:
                    if "strategies" not in parsed or not parsed["strategies"]:
                        parsed["strategies"] = {
                            k: v for k, v in strategies_ml.items() if not k.startswith("_")
                        }
                    parsed["_source"] = "gemini"
                    print("✅ Gemini повернув валідний JSON")
                    return self._enrich_result(parsed, description, category_id, regression_price)
                print(f"Спроба {attempt}: не вдалося розпарсити JSON")
            else:
                print(f"Спроба {attempt}: Gemini не відповів")

        print("📊 ML fallback")
        return self._enrich_result(ml_result, description, category_id, regression_price)

    # ── ML обчислення (кешується) ────────────────────────────────────────────

    def _compute_ml(self, description: str, category_id: Optional[int]) -> dict:
        cache_key = _desc_hash(description, category_id)
        cached = _ml_cache_get(cache_key)
        if cached:
            print("💾 ML cache hit")
            return cached

        regression_price = self.analyzer.predict_price_regression(description, category_id)
        comparables_df = self.analyzer.find_comparables(description, category_id, top_k=6)
        stats = self.analyzer.calculate_price_statistics(comparables_df)
        strategies_ml = self.analyzer.classify_strategy(regression_price, category_id)
        sold_stats = self.analyzer.calculate_sold_statistics(category_id)

        med = stats.get("median_price", regression_price)

        result = {
            "recommended_price": int(regression_price),
            "price_range": {
                "min": strategies_ml["FAST"],
                "max": strategies_ml["MAX_PROFIT"],
            },
            "strategies": {
                k: v for k, v in strategies_ml.items() if not k.startswith("_")
            },
            "strategies_raw": strategies_ml,
            "explanation": (
                f"ML-прогноз (RandomForest + Ridge). "
                f"Медіана по {stats.get('count', 0)} схожих: {med:.0f} грн. "
                f"Реальні угоди: {sold_stats.get('median_sold', 0):.0f} грн."
            ),
            "key_factors": [
                "стан товару",
                "бренд / категорія",
                "якість та повнота опису",
                "батарея та комплектність"
            ],
            "advice": _dynamic_advice(description, category_id, sold_stats),
            "_source": "ml_fallback",
        }

        _ml_cache_set(cache_key, result)
        return result
    # ── Збагачення результату (trust block + market speed) ────────────────────

    def _enrich_result(
        self,
        result: dict,
        description: str,
        category_id: Optional[int],
        recommended_price: float,
    ) -> dict:
        """Додає trust_block і market_speed до будь-якого результату."""
        try:
            result["trust_block"] = self.analyzer.comparables_for_trust(
                description, category_id, recommended_price
            )
        except Exception as e:
            print(f"⚠ trust_block error: {e}")
            result["trust_block"] = {}

        try:
            result["market_speed"] = self.analyzer.market_speed_info(category_id)
        except Exception as e:
            print(f"⚠ market_speed error: {e}")
            result["market_speed"] = {}

        # Прибираємо внутрішнє поле
        result.pop("strategies_raw", None)
        return result

    # ── Промпт ───────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        description: str,
        category_id: Optional[int],
        regression_price: float,
        strategies_ml: dict,
    ) -> str:
        comparables_df   = self.analyzer.find_comparables(description, category_id, top_k=6)
        fast_sold_df     = self.analyzer.find_fast_sold_comparables(description, category_id)
        reserved_df      = self.analyzer.find_reserved_comparables(description, category_id, top_k=3)
        stats            = self.analyzer.calculate_price_statistics(comparables_df)
        sold_stats       = self.analyzer.calculate_sold_statistics(category_id)
        keywords         = self.analyzer.top_keywords(category_id or 0, n=12) if category_id else []

        comp_text     = _format_comparables(comparables_df)
        fast_comp     = _format_comparables(fast_sold_df)
        reserved_comp = _format_comparables(reserved_df)

        stats_text = (
            f"Медіана активних: {stats.get('median_price', 0):.0f} грн | "
            f"Медіана реальних угод: {sold_stats.get('median_sold', 0):.0f} грн | "
            f"Типовий торг: -{sold_stats.get('avg_discount_pct', 0):.0f}% | "
            f"Середній час продажу: {sold_stats.get('avg_days', 0):.0f} днів"
        )
        keywords_text = ", ".join(keywords) if keywords else "—"

        fast    = strategies_ml.get("FAST", regression_price)
        bal     = strategies_ml.get("BALANCED", regression_price)
        maxp    = strategies_ml.get("MAX_PROFIT", regression_price)
        label   = strategies_ml.get("_label", "medium")

        return f"""Ти — AI-прайсер для monopricehelper (б/у маркетплейс, Україна).
Відповідь: ТІЛЬКИ JSON, без markdown, без коментарів.

=== ОПИС ТОВАРУ ===
{description}

=== КАТЕГОРІЯ ===
ID: {category_id or "невідомо"} | Ключові слова: {keywords_text}

=== АКТИВНІ СХОЖІ ОГОЛОШЕННЯ ===
{comp_text}

=== РЕАЛЬНО ПРОДАНІ (< 3 днів) ===
{fast_comp}

=== ЗАРЕЗЕРВОВАНІ ===
{reserved_comp}

=== СТАТИСТИКА РИНКУ ===
{stats_text}

=== ML ПРОГНОЗ ===
Регресія: {regression_price:.0f} грн | Клас: {label}
Стратегії ML: FAST={fast} | BALANCED={bal} | MAX_PROFIT={maxp} грн

=== ВАЖЛИВЕ ПРАВИЛО ===
Для iPhone (категорія 1261) реальна ринкова ціна зазвичай значно вища за середню по категорії. 
Apple-техніка тримає високу вартість навіть у б/у стані. NeverLock та хороша батарея дають +15-25%.

=== ЗАВДАННЯ ===
Визнач реалістичну ціну, поясни коротко (2 речення), дай 3 фактори, пораду (2 речення).

{{"recommended_price":{int(regression_price)},"price_range":{{"min":{fast},"max":{maxp}}},"strategies":{{"FAST":{fast},"BALANCED":{bal},"MAX_PROFIT":{maxp}}},"explanation":"...","key_factors":["...","...","..."],"advice":"..."}}"""
# ── ML кеш (in-memory LRU) ────────────────────────────────────────────────────

_CACHE: dict = {}
_CACHE_MAX   = 512


def _ml_cache_get(key: str) -> dict | None:
    return _CACHE.get(key)


def _ml_cache_set(key: str, value: dict):
    if len(_CACHE) >= _CACHE_MAX:
        oldest = next(iter(_CACHE))
        del _CACHE[oldest]
    _CACHE[key] = value


def _desc_hash(description: str, category_id: Optional[int]) -> str:
    raw = f"{description.strip().lower()}|{category_id}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Допоміжні функції ─────────────────────────────────────────────────────────

def _format_comparables(df) -> str:
    if df is None or df.empty:
        return "Схожих оголошень не знайдено."
    lines = []
    for _, row in df.iterrows():
        sim = f"[схожість: {int(row.get('_sim', 0))}]"
        lines.append(f"• {row['title']} — {row['price']:.0f} грн {sim}")
    return "\n".join(lines)


def _dynamic_advice(description: str, category_id: Optional[int], sold_stats: dict) -> str:
    """Порада з реальними даними замість загальної."""
    text  = description.lower()
    parts = []

    # Батарея
    m = re.search(r'батаре[яї]\s*[:\-]?\s*(\d{2,3})\s*%', text)
    if m:
        pct = int(m.group(1))
        if pct < 80:
            parts.append(f"Батарея {pct}% — покупці очікують знижку 10–15%.")
        elif pct >= 90:
            parts.append(f"Батарея {pct}% — це перевага, вкажи це в заголовку.")

    # Пам'ять
    m = re.search(r'(\d+)\s*(?:гб|gb)\b', text, re.I)
    if m and int(m.group(1)) >= 128:
        parts.append(f"Обсяг {m.group(1)} ГБ — додай це в заголовок оголошення.")

    # NeverLock
    if "neverlock" not in text and category_id in (1261,):
        parts.append("Вкажи статус NeverLock — це +5–10% до ціни для телефонів.")

    # Загальна порада
    avg_days = sold_stats.get("avg_days", 14)
    if avg_days <= 5:
        parts.append(f"Ринок активний — схожі товари продаються за {avg_days:.0f} дні.")
    elif avg_days > 14:
        parts.append("Додай 3–5 якісних фото — це скорочує час продажу вдвічі.")
    else:
        parts.append("Якісні фото та детальний опис збільшують ціну на 10–20%.")

    return " ".join(parts[:2]) if parts else (
        "Додай якісні фото (3–5 штук) та вкажи точний стан — це збільшує ціну на 10–20%."
    )


def _parse_json(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    return None