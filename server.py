"""
server.py — Flask сервер, що підключає веб-інтерфейс до PriceAgent

Запуск:
    pip install flask pillow
    python server.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import os

from agent import PriceAgent

app = Flask(__name__, static_folder="static")
CORS(app)  # дозволяємо CORS для локальної розробки

# ── Ініціалізуємо агент один раз при старті ────────────────────────────────
print("Завантажуємо PriceAgent...")
agent = PriceAgent()
print("✅ PriceAgent готовий!")


# ── Роут: головна сторінка ─────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ── Роут: API endpoint для аналізу ────────────────────────────────────────
@app.route("/api/price", methods=["POST"])
def price():
    try:
        # Отримуємо дані з форми
        description = request.form.get("description", "").strip()
        if not description:
            return jsonify({"error": "Опис товару обов'язковий"}), 400

        category_id = request.form.get("category_id")
        category_id = int(category_id) if category_id else None

        # Обробляємо фото якщо є
        images = []
        for file in request.files.getlist("images"):
            if file and file.filename:
                img_bytes = file.read()
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(img)

        # Запускаємо агент
        result = agent.price_item(
            description=description,
            images=images if images else None,
            category_id=category_id,
        )

        return jsonify(result)

    except Exception as e:
        print(f"❌ Помилка в /api/price: {e}")
        return jsonify({"error": str(e)}), 500


# ── Запуск ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8080, host="0.0.0.0")