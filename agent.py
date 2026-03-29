"""
agent.py — PriceAgent
Повний пайплайн:
  1. Регресія ціни (RF + Ridge ансамбль)
  2. Класифікація стратегії (RF classifier ? low/medium/high ? FAST/BALANCED/MAX_PROFIT)
  3. Пошук компаративів (keyword overlap)
  4. Ключові слова категорії в промпт
  5. Структурований JSON промпт
  6. Gemini 2.5 Flash (JSON mode)
  7. Парсинг і валідація JSON
  8. ML fallback якщо Gemini не відповів
"""

import json
import re
from typing import Optional

from utils.data_loader   import load_dataset
from utils.price_analyzer import PriceAnalyzer
from utils.gemini_client  import get_gemini_response


class PriceAgent:

    def __init__(self):
        self.df, fast_sold = load_dataset()
        self.analyzer = PriceAnalyzer(self.df, fast_sold)

    # ?? Головний метод ????????????????????????????????????????????????????????

    def price_item(
        self,
        description: str,
        images=None,
        category_id: Optional[int] = None,
        debug: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        description : текст опису від продавця
        images      : список Pillow Image або None
        category_id : ID категорії (опційно)
        debug       : друкувати сирий Gemini response

        Returns
        -------
        dict з ключами: recommended_price, price_range, strategies,
                        explanation, key_factors, advice, _source
        """

        # ?? 1. ML частина ?????????????????????????????????????????????????????
        regression_price = self.analyzer.predict_price_regression(description, category_id)
        comparables_df   = self.analyzer.find_comparables(description, category_id, top_k=6)
        stats            = self.analyzer.calculate_price_statistics(comparables_df)
        keywords         = self.analyzer.top_keywords(category_id or 0, n=12) if category_id else []
        strategies_ml = self.analyzer.classify_strategy(
            regression_price,
            category_id=category_id,
        )
        fast_sold_df = self.analyzer.find_fast_sold_comparables(description, category_id)
        sold_stats = self.analyzer.calculate_sold_statistics(category_id)

        # ?? 2. Формуємо контекст для промпту ??????????????????????????????????
        comp_text = _format_comparables(comparables_df)
        fast_comp_text = _format_comparables(fast_sold_df)
        stats_text = (
            f"Медіана активних: {stats.get('median_price', 0):.0f} грн | "
            f"Медіана реальних угод: {sold_stats.get('median_sold', 0):.0f} грн | "  # ← нове
            f"Типовий торг: -{sold_stats.get('avg_discount_pct', 0):.0f}% | "         # ← нове
            f"Середній час продажу: {sold_stats.get('avg_days', 0):.0f} днів"         # ← нове
        )
        keywords_text = ", ".join(keywords) if keywords else "—"

        # ?? 3. Промпт ?????????????????????????????????????????????????????????
        prompt = _build_prompt(
            description      = description,
            category_id      = category_id,
            comp_text        = comp_text,
            fast_comp_text   = fast_comp_text,
            stats_text       = stats_text,
            keywords_text    = keywords_text,
            regression_price = int(regression_price),
            strategies_ml    = strategies_ml,
        )

        print("Відправляємо в Gemini...")
        result = None
        for attempt in range(1, 3):
            print(f"⏳ Gemini, спроба {attempt}...")
            raw = get_gemini_response(prompt, images, debug=debug)
            if raw:
                result = _parse_json(raw)
                if result:
                    if "strategies" not in result or not result["strategies"]:
                        result["strategies"] = {
                            k: v for k, v in strategies_ml.items() if not k.startswith("_")
                        }
                    result["_source"] = "gemini"
                    print("Gemini повернув валідний JSON")
                    return result
                print(f"Спроба {attempt}: не вдалося розпарсити JSON")
            else:
                print(f"Спроба {attempt}: Gemini не відповів")

        # Якщо дійшли сюди - обидві спроби провалились
        print("ML fallback")
        med = stats.get("median_price", regression_price)
        return {
            "recommended_price": int(regression_price),
            "price_range": {
                "min": int(regression_price * 0.75),
                "max": int(regression_price * 1.35),
            },
            "strategies": {
                k: v for k, v in strategies_ml.items() if not k.startswith("_")
            },
            "explanation": (
                f"ML-прогноз на основі RandomForest. "
                f"Медіана по {stats.get('count', 0)} схожих оголошень: {med:.0f} грн."
            ),
            "key_factors": [
                "стан товару (умови)",
                "довжина та якість опису",
                "бренд / категорія",
            ],
            "advice": (
                "Додайте якісні фото (3–5 штук) та вкажіть точний стан товару — "
                "це збільшує ціну на 10–20%."
            ),
            "_source": "ml_fallback",
        }


# ?? Допоміжні функції ?????????????????????????????????????????????????????????

def _format_comparables(df) -> str:
    if df is None or df.empty:
        return "Схожих оголошень не знайдено."
    lines = []
    for _, row in df.iterrows():
        sim = f"[схожість: {int(row.get('_sim', 0))}]"
        lines.append(f"• {row['title']} — {row['price']:.0f} грн {sim}")
    return "\n".join(lines)


def _build_prompt(
    description, category_id, comp_text, stats_text, fast_comp_text,
    keywords_text, regression_price, strategies_ml,
) -> str:
    fast    = strategies_ml.get("FAST", regression_price)
    bal     = strategies_ml.get("BALANCED", regression_price)
    maxp    = strategies_ml.get("MAX_PROFIT", regression_price)
    label   = strategies_ml.get("_label", "medium")

    return f"""Ти — AI-прайсер для monoбазар (б/у маркетплейс, Україна).
Твоя задача: визначити справедливу ціну товару на основі ринкових даних.

=== ОПИС ТОВАРУ ===
{description}

=== КАТЕГОРІЯ ===
ID: {category_id or "невідомо"}
Топ ключових слів категорії: {keywords_text}

=== СХОЖІ ОГОЛОШЕННЯ ===
{comp_text}

=== РЕАЛЬНО ПРОДАНІ (< 3 днів) ===
{fast_comp_text}
↑ Це найсильніший сигнал — ці товари продались швидко за цю ціну.

=== СТАТИСТИКА РИНКУ ===
{stats_text}

=== ML ПРОГНОЗ ===
Регресія ціни: {regression_price} грн
Клас цінового рівня (ML): {label}
ML стратегії: FAST={fast} | BALANCED={bal} | MAX_PROFIT={maxp} грн

=== ЗАВДАННЯ ===
1. Визнач рекомендовану ціну та діапазон (min–max).
2. Поясни чому саме така ціна (посилайся на компаративи).
3. Вкажи 3 ключові фактори що впливають на ціну.
4. Дай коротку пораду продавцю.

Поверни ТІЛЬКИ JSON об'єкт (без markdown, без коментарів):
{{
  "recommended_price": {regression_price},
  "price_range": {{"min": {fast}, "max": {maxp}}},
  "strategies": {{"FAST": {fast}, "BALANCED": {bal}, "MAX_PROFIT": {maxp}}},
  "explanation": "...",
  "key_factors": ["...", "...", "..."],
  "advice": "..."
}}"""


def _parse_json(raw: str) -> dict | None:
    """Жадібний regex + кілька спроб парсингу."""
    # Спроба 1: прямий парсинг
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Спроба 2: витягти перший повний JSON об'єкт (жадібно)
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Спроба 3: прибрати markdown-огорожу
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return None
