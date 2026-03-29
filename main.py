"""
main.py — Запуск агента з прикладами
"""

import json
from agent import PriceAgent

agent = PriceAgent()
print("\n" + "=" * 70)

# ── Приклад 1: Книга (категорія 795) ─────────────────────────────────────────
print("📚 ПРИКЛАД 1: Книга")
r1 = agent.price_item(
    description="""
    Книга "Portraits" Джона Бергера про художників у новому стані. 
    Видавництво Verso, редактор Tom Overton. Ідеальний новий стан, без нюансів.
    Відправка Новою Поштою за передоплатою 300 грн.
    """,
    category_id=795,
)
print(json.dumps(r1, indent=2, ensure_ascii=False))

print("\n" + "=" * 70)

# ── Приклад 2: Кросівки (категорія 512) ──────────────────────────────────────
print("👟 ПРИКЛАД 2: Кросівки")
r2 = agent.price_item(
    description="""
    Кросівки Saucony Triumph 17, чоловічі. Розмір EU 42 / US 8.5.
    Синьо-чорний колір. Стан: нові, не ношені. Оригінал.
    Куплені в США, є чек. Відправка після передоплати.
    """,
    category_id=512,
)
print(json.dumps(r2, indent=2, ensure_ascii=False))

print("\n" + "=" * 70)

# ── Приклад 3: Телефон (без категорії) ───────────────────────────────────────
print("📱 ПРИКЛАД 3: Телефон")
r3 = agent.price_item(
    description="""
    iPhone 13 Pro 256GB Sierra Blue. Стан гарний, є невеликі подряпини на корпусі.
    Комплект: коробка, кабель. Батарея 87%. Face ID працює ідеально.
    """
)
print(json.dumps(r3, indent=2, ensure_ascii=False))

print("\n" + "=" * 70)
print("✅ Готово!")
