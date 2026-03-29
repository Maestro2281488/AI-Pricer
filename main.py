"""
main.py — Запуск агента з прикладами

Демонструє:
- fast_mode=True (паралельний ML+Gemini, < 5 сек)
- trust_block (блок довіри для UI)
- market_speed (індикатор швидкості ринку)
- динамічна порада
"""

import json
import time
from agent import PriceAgent

agent = PriceAgent()
print("\n" + "=" * 70)


def run_example(label: str, description: str, category_id=None):
    print(f"\n{label}")
    t0 = time.time()
    result = agent.price_item(
        description=description,
        category_id=category_id,
        fast_mode=True,
        debug=False,
    )
    elapsed = time.time() - t0
    print(f"⏱ Час: {elapsed:.2f}с | Джерело: {result.get('_source', '?')}")

    # Основний результат
    core = {k: v for k, v in result.items() if k not in ("trust_block", "market_speed")}
    print(json.dumps(core, indent=2, ensure_ascii=False))

    # Блок довіри (скорочено)
    tb = result.get("trust_block", {})
    if tb:
        print("\n📊 БЛОК ДОВІРИ:")
        for section, data in tb.items():
            if section == "price_position":
                pos = data
                if pos:
                    print(
                        f"  Позиція ціни: {pos.get('label','')} "
                        f"({pos.get('percentile',0):.0f}й перцентиль) "
                        f"— {pos.get('recommendation','')}"
                    )
            else:
                items = data.get("items", [])
                stats = data.get("stats", {})
                if items:
                    print(f"\n  {data.get('label', section)}:")
                    for item in items[:3]:
                        print(f"    • {item['title']} — {item['price']:.0f} грн")
                    if stats.get("count", 0):
                        print(
                            f"    Медіана: {stats['median_price']:.0f} грн "
                            f"(N={stats['count']})"
                        )

    # Швидкість ринку
    ms = result.get("market_speed", {})
    if ms:
        print(f"\n{ms.get('speed_label','')}: продаються за ~{ms.get('avg_days_to_sell',0):.0f} днів "
              f"| Активних оголошень: {ms.get('active_listings', 0)}")

    print("\n" + "=" * 70)


# ── Приклад 1: Книга (категорія 795) ─────────────────────────────────────────
run_example(
    "📚 ПРИКЛАД 1: Книга",
    """
    Книга "Portraits" Джона Бергера про художників у новому стані.
    Видавництво Verso, редактор Tom Overton. Ідеальний новий стан, без нюансів.
    Відправка Новою Поштою за передоплатою 300 грн.
    """,
    category_id=795,
)

# ── Приклад 2: Кросівки (категорія 512) ──────────────────────────────────────
run_example(
    "👟 ПРИКЛАД 2: Кросівки",
    """
    Кросівки Saucony Triumph 17, чоловічі. Розмір EU 42 / US 8.5.
    Синьо-чорний колір. Стан: нові, не ношені. Оригінал.
    Куплені в США, є чек. Відправка після передоплати.
    """,
    category_id=512,
)

# ── Приклад 3: Телефон (категорія 1261) ──────────────────────────────────────
run_example(
    "📱 ПРИКЛАД 3: Телефон",
    """
    iPhone 13 Pro 256GB Sierra Blue. Стан гарний, є невеликі подряпини на корпусі.
    Комплект: коробка, кабель. Батарея 87%. Face ID працює ідеально. NeverLock.
    """,
    category_id=1261,
)

print("✅ Готово!")