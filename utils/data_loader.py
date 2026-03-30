"""
utils/data_loader.py
Оновлено під актуальний файл hackaton_advertisements_with_id.csv (з 10 колонками)
"""

import os
import pandas as pd
import numpy as np
import re
from collections import Counter

CONDITION_GOOD = [
    "новий", "нова", "нове", "new", "ідеальний", "ідеальна", "ідеальне",
    "ідеал", "без нюансів", "без пошкоджень", "не використовувався",
    "не ношений", "не ношена", "запакований", "в упаковці", "sealed",
    "mint", "perfect", "відмінний стан", "відмінна"
]

CONDITION_BAD = [
    "б/у", "вживаний", "вживана", "вживане", "подряпини", "потертості",
    "пошкодження", "зламаний", "зламана", "не працює", "на запчастини",
    "тріщина", "скол", "дефект", "worn", "used", "damaged"
]

PRICE_QUANTILES = (0.33, 0.66)


def load_dataset(path: str = "data/hackaton_advertisements_with_id.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не знайдено: {path}")

    # Завантажуємо актуальний файл (з заголовками)
    df = pd.read_csv(path, low_memory=False)

    # Очищаємо пробіли в назвах колонок
    df.columns = df.columns.str.strip()

    # Підлаштовуємо під очікування коду тіммейта (id замість advertisement_id)
    if 'advertisement_id' in df.columns:
        df.rename(columns={'advertisement_id': 'id'}, inplace=True)

    print(f"✅ Завантажено {len(df):,} рядків з актуального файлу")

    # Конвертація типів
    df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce")

    if "sold_price" in df.columns:
        df["sold_price"] = pd.to_numeric(df["sold_price"], errors="coerce")
    else:
        df["sold_price"] = np.nan

    df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["modified_at"] = pd.to_datetime(df["modified_at"], errors="coerce")

    # Основна ціна для ML: sold_price якщо є, інакше original_price
    df["price"] = df["sold_price"].fillna(df["original_price"])

    total = len(df)
    active = (df["status"] == "ACTIVE").sum()
    print(f"✅ Завантажено {total:,} оголошень | Активних: {active:,}")

    fast_sold = _load_fast_sold(df)
    reserved_df = _load_reserved(df)
    deleted_df = _load_deleted(df)

    # Фільтруємо тільки активні оголошення для навчання
    df = df[
        (df["status"] == "ACTIVE") &
        df["price"].notna() &
        (df["price"] > 0)
    ].copy().reset_index(drop=True)

    _enrich(df)

    print(f"✅ Розмічено: {len(df):,} активних оголошень")
    print(f"   price_label:\n{df['price_label'].value_counts().to_string()}")
    print(f"   RESERVED/ORDER_PROCESSING: {len(reserved_df):,}")
    print(f"   DELETED (негативний сигнал): {len(deleted_df):,}")

    return df, fast_sold, reserved_df, deleted_df


def _enrich(df: pd.DataFrame):
    """Додає всі фічі in-place."""
    df["text"] = (
        df["title"].fillna("") + " " + df["description"].fillna("")
    ).str.lower()

    df["condition_score"] = df["text"].apply(_condition_score)
    df["desc_len"] = df["description"].fillna("").str.len()
    df["title_len"] = df["title"].fillna("").str.len()
    df["has_brand"] = df["text"].apply(_has_brand).astype(int)

    # Відновлюємо колонки торгу, оскільки новий датасет їх має
    if "sold_via_bargain" in df.columns:
        df["was_bargained"] = df["sold_via_bargain"].fillna(False).astype(int)
    else:
        df["was_bargained"] = 0

    if "sold_price" in df.columns:
        df["price_discount"] = (
            (df["original_price"] - df["sold_price"]) / df["original_price"].replace(0, np.nan)
        ).fillna(0).clip(0, 1)
    else:
        df["price_discount"] = 0.0

    df["keyword_score"] = _keyword_score_per_category(df)
    df["price_label"] = _label_price(df)

    # Recency decay
    now = pd.Timestamp.now(tz="UTC")
    days_old = (now - df["created_at"]).dt.days.clip(0, 365).fillna(180)
    df["recency_weight"] = np.exp(-days_old / 30)

    # Окрема медіана проданих (використовуємо price, який вже враховує sold_price)
    sold_med = (
        df.groupby("category_id")["price"]
        .transform("median")
        .fillna(df["price"].median())
    )
    df["log_cat_sold_median"] = np.log1p(sold_med)

    # Regex-витяг характеристик
    df["battery_pct"] = df["text"].apply(_extract_battery)
    df["memory_gb"] = df["text"].apply(_extract_memory)
    df["item_year"] = df["text"].apply(_extract_year)

    df["has_specs"] = (
        df["battery_pct"].notna() |
        df["memory_gb"].notna() |
        df["item_year"].notna()
    ).astype(int)


def _load_fast_sold(df: pd.DataFrame) -> pd.DataFrame:
    """SOLD товари що продались за < 3 днів."""
    d = df.copy()
    d["price"] = d["sold_price"].fillna(d["original_price"])

    cutoff = pd.Timestamp("2026-02-27", tz="UTC")
    days_to_sell = (d["modified_at"] - d["created_at"]).dt.days

    fast = d[
        (d["status"] == "SOLD") &
        (d["created_at"] >= cutoff) &
        (
            (days_to_sell < 3) |
            (d["category_id"].isin([1261, 1320]))
        ) &
        d["price"].notna()
    ].copy()

    fast["text"] = (
        fast["title"].fillna("") + " " + fast["description"].fillna("")
    ).str.lower()

    print(f"✅ Швидко продані (SOLD < 3 дні): {len(fast):,}")
    return fast


def _load_reserved(df: pd.DataFrame) -> pd.DataFrame:
    reserved = df[
        df["status"].isin(["RESERVED", "ORDER_PROCESSING"]) &
        df["original_price"].notna() &
        (df["original_price"] > 0)
    ].copy()

    reserved["price"] = reserved["original_price"]
    reserved["text"] = (
        reserved["title"].fillna("") + " " + reserved["description"].fillna("")
    ).str.lower()

    now = pd.Timestamp.now(tz="UTC")
    days_old = (now - reserved["created_at"]).dt.days.clip(0, 365).fillna(90)
    reserved["recency_weight"] = np.exp(-days_old / 30)

    print(f"✅ RESERVED/ORDER_PROCESSING (ціна спрацювала): {len(reserved):,}")
    return reserved


def _load_deleted(df: pd.DataFrame) -> pd.DataFrame:
    deleted = df[
        (df["status"] == "DELETED") &
        df["original_price"].notna() &
        (df["original_price"] > 0)
    ].copy()

    deleted["price"] = deleted["original_price"]
    deleted["is_overpriced"] = 1
    deleted["text"] = (
        deleted["title"].fillna("") + " " + deleted["description"].fillna("")
    ).str.lower()

    print(f"✅ DELETED без продажу (завищена ціна): {len(deleted):,}")
    return deleted


# ── Regex функції ─────────────────────────────────────────────────────────────

def _extract_battery(text: str):
    m = re.search(r'батаре[яї]\s*[:\-]?\s*(\d{2,3})\s*%', text)
    if not m:
        m = re.search(r'(\d{2,3})\s*%\s*(?:батаре[яї]|заряд)', text)
    return float(m.group(1)) if m else np.nan


def _extract_memory(text: str):
    m = re.search(r'(\d+)\s*/\s*(\d+)\s*(?:гб|gb)\b', text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r'(\d+)\s*(?:гб|gb)\b', text, re.I)
    return float(m.group(1)) if m else np.nan


def _extract_year(text: str):
    m = re.search(r'\b(201[5-9]|202[0-6])\b', text)
    return float(m.group(1)) if m else np.nan


def _condition_score(text: str) -> float:
    good = sum(1 for w in CONDITION_GOOD if w in text)
    bad = sum(1 for w in CONDITION_BAD if w in text)
    if good == 0 and bad == 0:
        return 0.5
    return round(good / (good + bad + 1e-9), 3)


def _has_brand(text: str) -> int:
    brands = r"nike|adidas|saucony|apple|samsung|levi|canon|sony|verso|iphone|ipad|macbook|xiaomi|puma|reebok|new balance|zara|h&m|mango|reserved|ikea|lg|bosch|philips"
    return int(bool(re.search(brands, text, re.I)))


def _keyword_score_per_category(df: pd.DataFrame) -> pd.Series:
    scores = pd.Series(0.5, index=df.index)
    for cat_id, group in df.groupby("category_id"):
        all_words = []
        for txt in group["text"]:
            all_words.extend(re.findall(r"[а-яёіїєa-z]{4,}", txt))
        if not all_words:
            continue
        top_words = set(w for w, _ in Counter(all_words).most_common(50))

        def score_row(txt, tw=top_words):
            words = set(re.findall(r"[а-яёіїєa-z]{4,}", txt))
            return round(len(words & tw) / len(words), 3) if words else 0.5

        result = {idx: score_row(txt) for idx, txt in zip(group.index, group["text"])}
        scores.update(pd.Series(result))
    return scores


def _label_price(df: pd.DataFrame) -> pd.Series:
    labels = pd.Series("medium", index=df.index)
    global_q = df["price"].quantile(PRICE_QUANTILES).values

    for cat_id, group in df.groupby("category_id"):
        if len(group) >= 5:
            q_lo, q_hi = group["price"].quantile(PRICE_QUANTILES).values
        else:
            q_lo, q_hi = global_q
        labels.loc[group[group["price"] <= q_lo].index] = "low"
        labels.loc[group[group["price"] >= q_hi].index] = "high"
    return labels