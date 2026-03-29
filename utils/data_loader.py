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


def load_dataset(path: str = "data/hackaton_advertisements_with_id.csv") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Нова схема колонок
    df.columns = df.columns.str.strip()

    df["original_price"]  = pd.to_numeric(df["original_price"], errors="coerce")
    df["sold_price"]      = pd.to_numeric(df["sold_price"],      errors="coerce")
    df["category_id"]     = pd.to_numeric(df["category_id"],     errors="coerce")
    df["created_at"]      = pd.to_datetime(df["created_at"],     errors="coerce")
    df["modified_at"]     = pd.to_datetime(df["modified_at"],    errors="coerce")

    # Основна ціна для ML: sold_price якщо є, інакше original_price
    df["price"] = df["sold_price"].fillna(df["original_price"])

    total  = len(df)
    active = (df["status"] == "ACTIVE").sum()
    print(f"✅ Завантажено {total:,} оголошень | Активних: {active:,}")

    fast_sold = _load_fast_sold(df)

    # Залишаємо тільки ACTIVE з валідною ціною
    df = df[
        (df["status"] == "ACTIVE") &
        df["price"].notna() &
        (df["price"] > 0)
    ].copy().reset_index(drop=True)

    # Розмітка
    df["text"] = (
        df["title"].fillna("") + " " + df["description"].fillna("")
    ).str.lower()

    df["condition_score"] = df["text"].apply(_condition_score)
    df["desc_len"]        = df["description"].fillna("").str.len()
    df["title_len"]       = df["title"].fillna("").str.len()
    df["has_brand"]       = df["text"].apply(_has_brand).astype(int)

    # Чи продавався зі знижкою (новий сигнал)
    df["was_bargained"]   = df["sold_via_bargain"].fillna(False).astype(int)

    # Дисконт = наскільки sold_price < original_price (0 якщо немає даних)
    df["price_discount"]  = (
        (df["original_price"] - df["sold_price"]) / df["original_price"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    df["keyword_score"]   = _keyword_score_per_category(df)
    df["price_label"]     = _label_price(df)

    print(f"✅ Розмічено: {len(df):,} активних оголошень")
    print(f"   price_label розподіл:\n{df['price_label'].value_counts().to_string()}")
    return df, fast_sold


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

    print(f"✅ Швидко продані: {len(fast):,} оголошень")
    return fast


def _condition_score(text: str) -> float:
    good = sum(1 for w in CONDITION_GOOD if w in text)
    bad  = sum(1 for w in CONDITION_BAD  if w in text)
    if good == 0 and bad == 0:
        return 0.5
    return round(good / (good + bad + 1e-9), 3)


def _has_brand(text: str) -> int:
    brands = (
        r"nike|adidas|saucony|apple|samsung|levi|canon|sony|verso|"
        r"iphone|ipad|macbook|xiaomi|puma|reebok|new balance|"
        r"zara|h&m|mango|reserved|ikea|lg|bosch|philips"
    )
    return int(bool(re.search(brands, text)))


def _keyword_score_per_category(df: pd.DataFrame) -> pd.Series:
    scores = pd.Series(0.5, index=df.index)
    result = {}
    for cat_id, group in df.groupby("category_id"):
        all_words = []
        for txt in group["text"]:
            all_words.extend(re.findall(r"[а-яёіїєa-z]{4,}", txt))
        if not all_words:
            continue
        top_words = set(w for w, _ in Counter(all_words).most_common(50))
        def score_row(txt):
            words = set(re.findall(r"[а-яёіїєa-z]{4,}", txt))
            return round(len(words & top_words) / len(words), 3) if words else 0.5
        result.update(dict(zip(group.index, group["text"].apply(score_row))))
    scores.update(pd.Series(result))
    return scores


def _label_price(df: pd.DataFrame) -> pd.Series:
    labels   = pd.Series("medium", index=df.index)
    global_q = df["price"].quantile(PRICE_QUANTILES).values

    for cat_id, group in df.groupby("category_id"):
        q_lo, q_hi = (
            group["price"].quantile(PRICE_QUANTILES).values
            if len(group) >= 5
            else global_q
        )
        labels.loc[group[group["price"] <= q_lo].index] = "low"
        labels.loc[group[group["price"] >= q_hi].index] = "high"
    return labels