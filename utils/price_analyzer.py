"""
utils/price_analyzer.py

Покращення:
- RESERVED/ORDER_PROCESSING як навчальний сигнал у регресії
- DELETED як негативний сигнал (штраф до ціни)
- Recency-зважена регресія (свіжі дані важливіші)
- log_cat_sold_median як окрема фіча
- battery_pct / memory_gb / item_year / has_specs у структурних фічах
- market_speed_info() — індикатор швидкості ринку для UI
- price_position() — де ціна відносно ринку (прогрес-бар для UI)
- comparables_for_trust() — компаративи для блоку довіри в UI
"""

import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional, Dict, List, Tuple

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_percentage_error
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings("ignore")


BRAND_RE = re.compile(
    r"nike|adidas|saucony|apple|samsung|levi|canon|sony|verso|"
    r"iphone|ipad|macbook|xiaomi|puma|reebok|new balance|"
    r"zara|h&m|mango|reserved|ikea|lg|bosch|philips",
    re.I,
)
CONDITION_GOOD_RE = re.compile(
    r"новий|нова|нове|ідеал|без нюансів|не ношен|запакован|sealed|mint", re.I
)
CONDITION_BAD_RE = re.compile(
    r"б/у|вживан|подряпин|потертост|пошкодж|зламан|не працює|тріщин|дефект|worn|damaged",
    re.I,
)


class PriceAnalyzer:

    def __init__(
        self,
        df: pd.DataFrame,
        fast_sold: pd.DataFrame = None,
        reserved_df: pd.DataFrame = None,   # НОВИЙ
        deleted_df: pd.DataFrame = None,     # НОВИЙ
    ):
        self.df          = df.copy()
        self.fast_sold   = fast_sold
        self.reserved_df = reserved_df if reserved_df is not None else pd.DataFrame()
        self.deleted_df  = deleted_df  if deleted_df  is not None else pd.DataFrame()

        self._build_features()
        self._train_regression()

    # ── Feature engineering ───────────────────────────────────────────────────

    def _build_features(self):
        d = self.df
        d["desc_len"]   = d["description"].fillna("").str.len()
        d["title_len"]  = d["title"].fillna("").str.len()
        d["word_count"] = d["description"].fillna("").str.split().str.len()
        d["has_brand"]  = d["text"].apply(lambda x: int(bool(BRAND_RE.search(x))))
        d["cond_good"]  = d["text"].apply(lambda x: int(bool(CONDITION_GOOD_RE.search(x))))
        d["cond_bad"]   = d["text"].apply(lambda x: int(bool(CONDITION_BAD_RE.search(x))))
        d["category_id"] = d["category_id"].fillna(0)

        cat_med = d.groupby("category_id")["price"].transform("median")
        d["cat_median"]     = cat_med.fillna(d["price"].median())
        d["log_cat_median"] = np.log1p(d["cat_median"])

        # Окремо медіана sold_price
        sold_med = d.groupby("category_id")["sold_price"].transform("median").fillna(
            d["price"].median()
        )
        d["log_cat_sold_median"] = np.log1p(sold_med)

        d["price_pct_rank"] = d.groupby("category_id")["price"].rank(pct=True)
        d["discount_rate"]  = (
            (d["original_price"] - d["sold_price"]) / d["original_price"]
        ).clip(0, 1).fillna(0)

        cat_discount = d.groupby("category_id")["discount_rate"].transform("median")
        d["cat_discount_median"] = cat_discount.fillna(0)

        d["days_to_sell"] = (d["modified_at"] - d["created_at"]).dt.days.clip(0, 365)
        cat_days = d.groupby("category_id")["days_to_sell"].transform("median")
        d["cat_days_median"] = cat_days.fillna(30)

        # Recency weight (вже може бути з data_loader, але перераховуємо для надійності)
        if "recency_weight" not in d.columns:
            now = pd.Timestamp.now(tz="UTC")
            days_old = (now - d["created_at"]).dt.days.clip(0, 365).fillna(180)
            d["recency_weight"] = np.exp(-days_old / 30)

        # Specs (вже можуть бути з data_loader)
        for col in ["battery_pct", "memory_gb", "item_year", "has_specs"]:
            if col not in d.columns:
                d[col] = 0

    def _structural_features(self) -> List[str]:
        return [
            "desc_len", "title_len", "has_brand",
            "cond_good", "cond_bad", "word_count",
            "category_id", "log_cat_median", "log_cat_sold_median",
            "condition_score", "keyword_score",
            "battery_pct", "memory_gb", "has_specs",
        ]

    def _get_struct_matrix(self, df_slice: pd.DataFrame) -> pd.DataFrame:
        return df_slice[self._structural_features()].fillna(0)

    # ── 1. Регресія ───────────────────────────────────────────────────────────

    def _train_regression(self):
        """
        Навчаємо на трьох джерелах з різними вагами:
          - sold_price (SOLD)          вага = 3.0 (найнадійніше)
          - RESERVED/ORDER_PROCESSING  вага = 2.0 (ціна спрацювала)
          - ACTIVE з original_price    вага = 1.0 (ринкова пропозиція)
          - DELETED (negative signal)  вага = 0.5, з penalty
        """
        frames = []

        # ── Продані (основне джерело) ──
        sold = self.df[self.df["sold_price"].notna()].copy()
        sold["_target"] = sold["sold_price"]
        sold["_weight"] = 3.0
        frames.append(sold)

        # ── RESERVED / ORDER_PROCESSING ──
        if not self.reserved_df.empty:
            res = self.reserved_df.copy()
            # Додаємо потрібні фічі якщо їх немає
            res = self._ensure_features(res)
            res["_target"] = res["original_price"]
            res["_weight"] = 2.0
            frames.append(res)

        # ── DELETED — таргет зі знижкою 15% (бо ціна була завищена) ──
        if not self.deleted_df.empty:
            del_ = self.deleted_df.copy()
            del_ = self._ensure_features(del_)
            del_["_target"] = del_["original_price"] * 0.85   # штраф 15%
            del_["_weight"] = 0.5
            frames.append(del_)

        train = pd.concat(frames, ignore_index=True)
        train = train[train["_target"].notna() & (train["_target"] > 0)]

        # Clip outliers
        q99 = train["_target"].quantile(0.99)
        train = train[train["_target"] <= q99]

        y       = np.log1p(train["_target"])
        weights = train["_weight"].values

        # TF-IDF + Ridge
        self.tfidf = TfidfVectorizer(
            max_features=3000, min_df=5,
            ngram_range=(1, 2), sublinear_tf=True,
        )
        text_col   = train["text"].fillna("") if "text" in train.columns else pd.Series([""] * len(train))
        X_tfidf    = self.tfidf.fit_transform(text_col)
        X_struct   = self._get_struct_matrix(train)
        X_combined = hstack([csr_matrix(X_struct.values), X_tfidf])

        self.ridge = Ridge(alpha=10.0)
        self.ridge.fit(X_combined, y, sample_weight=weights)

        # RF (тільки структурні фічі)
        self.rf = RandomForestRegressor(
            n_estimators=120, max_depth=12,
            min_samples_leaf=5, random_state=42, n_jobs=-1,
        )
        self.rf.fit(X_struct, y, sample_weight=weights)

        # Метрика
        y_pred_log = 0.6 * self.ridge.predict(X_combined) + 0.4 * self.rf.predict(X_struct)
        y_pred     = np.expm1(y_pred_log)
        y_real     = np.expm1(y)
        mape = mean_absolute_percentage_error(y_real, y_pred) * 100
        n    = len(train)
        print(f"✅ Регресія навчена | Train MAPE: {mape:.1f}% | N={n:,} (sold+reserved+deleted)")

    def _ensure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Додає потрібні фічі якщо відсутні (для reserved/deleted df)."""
        df = df.copy()
        if "text" not in df.columns:
            df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()
        df["desc_len"]   = df["description"].fillna("").str.len()
        df["title_len"]  = df["title"].fillna("").str.len()
        df["word_count"] = df["description"].fillna("").str.split().str.len()
        df["has_brand"]  = df["text"].apply(lambda x: int(bool(BRAND_RE.search(x))))
        df["cond_good"]  = df["text"].apply(lambda x: int(bool(CONDITION_GOOD_RE.search(x))))
        df["cond_bad"]   = df["text"].apply(lambda x: int(bool(CONDITION_BAD_RE.search(x))))
        df["category_id"] = df["category_id"].fillna(0)

        global_med = self.df["price"].median()
        cat_med    = self.df.groupby("category_id")["price"].median()
        df["cat_median"] = df["category_id"].map(cat_med).fillna(global_med)
        df["log_cat_median"] = np.log1p(df["cat_median"])

        sold_med = self.df.groupby("category_id")["sold_price"].median()
        df["log_cat_sold_median"] = np.log1p(df["category_id"].map(sold_med).fillna(global_med))

        for col in ["condition_score", "keyword_score"]:
            if col not in df.columns:
                df[col] = 0.5
        for col in ["battery_pct", "memory_gb", "has_specs"]:
            if col not in df.columns:
                df[col] = 0
        return df

    # ── 2. Predict ────────────────────────────────────────────────────────────

    def predict_price_regression(
        self,
        description: str,
        category_id: Optional[int] = None,
    ) -> float:
        cat_id = category_id or 0
        cat_med = (
            self.df[self.df["category_id"] == cat_id]["price"].median()
            if cat_id in self.df["category_id"].values
            else self.df["price"].median()
        )
        sold_med = cat_med  # оскільки sold_price майже немає

        text = description.lower()
        row = {
            "desc_len": len(description),
            "title_len": len(description.split()[:15]) * 6,
            "has_brand": int(bool(BRAND_RE.search(text))),
            "cond_good": int(bool(CONDITION_GOOD_RE.search(text))),
            "cond_bad": int(bool(CONDITION_BAD_RE.search(text))),
            "word_count": len(description.split()),
            "category_id": cat_id,
            "log_cat_median": np.log1p(cat_med),
            "log_cat_sold_median": np.log1p(sold_med),
            "condition_score": _cond_score(text),
            "keyword_score": 0.5,
            "battery_pct": _extract_num(text, r'батаре[яї]\s*[:\-]?\s*(\d{2,3})\s*%', 0),
            "memory_gb": _extract_num(text, r'(\d+)\s*(?:гб|gb)\b', 0),
            "has_specs": int(bool(re.search(r'\d+\s*(?:gb|гб|%)', text, re.I))),
        }
        X_struct = pd.DataFrame([row])
        X_tfidf = self.tfidf.transform([text])
        X_combined = hstack([csr_matrix(X_struct.values), X_tfidf])

        ridge_log = float(self.ridge.predict(X_combined)[0])
        rf_log = float(self.rf.predict(X_struct)[0])
        pred_log = 0.6 * ridge_log + 0.4 * rf_log
        pred = float(np.expm1(pred_log))

        # === ВАЖЛИВІ КОРЕКЦІЇ ===
        if cat_id == 1261:  # Телефони
            pred = pred * 1.55   # сильний буст для Apple
            # Мінімальний поріг
            min_price = max(3200, cat_med * 0.75)
            pred = max(pred, min_price)

        return max(50.0, round(pred / 50) * 50)

    # ── 3. Стратегія ─────────────────────────────────────────────────────────

    def classify_strategy(self, recommended_price, category_id=None):
        cat_id = category_id or 0

        # Базова вибірка: sold + reserved (обидва сигналізують "ціна ОК")
        sold = self.df[
            (self.df["category_id"] == cat_id) &
            self.df["sold_price"].notna()
        ]["sold_price"]

        reserved_prices = pd.Series(dtype=float)
        if not self.reserved_df.empty and "original_price" in self.reserved_df.columns:
            mask = self.reserved_df["category_id"] == cat_id
            reserved_prices = self.reserved_df[mask]["original_price"].dropna()

        # Об'єднуємо sold і reserved з вагою
        all_success = pd.concat([sold, reserved_prices])
        prices = all_success if len(all_success) >= 10 else \
                 self.df[self.df["category_id"] == cat_id]["price"]

        if prices.empty:
            pct = 0.5
        else:
            pct = float((prices < recommended_price).mean())

        # Середній дисконт по категорії
        cat_discount = self.df[
            (self.df["category_id"] == cat_id) &
            self.df["discount_rate"].notna()
        ]["discount_rate"].median()
        cat_discount = cat_discount if not np.isnan(cat_discount) else 0.10

        fast_discount = max(0.10, cat_discount * 1.5)
        max_premium   = 0.25 if pct > 0.65 else 0.20

        return {
            "FAST":       max(50, int(round(recommended_price * (1 - fast_discount) / 50) * 50)),
            "BALANCED":   max(50, int(round(recommended_price / 50) * 50)),
            "MAX_PROFIT": max(50, int(round(recommended_price * (1 + max_premium) / 50) * 50)),
            "_label":     "high" if pct > 0.65 else "medium" if pct > 0.35 else "low",
            "_pct_rank":  round(pct, 2),
        }

    # ── 4. Компаративи ────────────────────────────────────────────────────────

    def find_comparables(
        self,
        query: str,
        category_id: Optional[int] = None,
        top_k: int = 8,
        source_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        df = (source_df if source_df is not None else self.df).copy()
        if category_id is not None:
            cat_df = df[df["category_id"] == category_id]
            df = cat_df if len(cat_df) >= 5 else df

        query_words = set(re.findall(r"[а-яёіїєa-z]{3,}", query.lower()))
        df["_sim"] = df["text"].apply(
            lambda t: len(query_words & set(re.findall(r"[а-яёіїєa-z]{3,}", t)))
        )
        top = df.nlargest(top_k, "_sim")
        if top["_sim"].max() > 0:
            top = top[top["_sim"] > 0]
        return top[["title", "price", "description", "category_id", "_sim"]].head(top_k)

    def find_fast_sold_comparables(self, query: str, category_id=None, top_k=5):
        if self.fast_sold is None or self.fast_sold.empty:
            return pd.DataFrame()
        return self.find_comparables(query, category_id, top_k, source_df=self.fast_sold)

    def find_reserved_comparables(self, query: str, category_id=None, top_k=5):
        """Компаративи з RESERVED — додатковий сигнал довіри."""
        if self.reserved_df.empty:
            return pd.DataFrame()
        return self.find_comparables(query, category_id, top_k, source_df=self.reserved_df)

    # ── 5. Статистика ─────────────────────────────────────────────────────────

    def calculate_price_statistics(self, comparables: pd.DataFrame) -> Dict:
        if comparables.empty or "price" not in comparables.columns:
            return {"avg_price": 0, "median_price": 0,
                    "min_price": 0, "max_price": 0, "count": 0}
        prices = comparables["price"].dropna()
        if prices.empty:
            return {"avg_price": 0, "median_price": 0,
                    "min_price": 0, "max_price": 0, "count": 0}
        q25, q75 = prices.quantile([0.25, 0.75])
        return {
            "avg_price":    round(float(prices.mean()),   0),
            "median_price": round(float(prices.median()), 0),
            "min_price":    round(float(prices.min()),    0),
            "max_price":    round(float(prices.max()),    0),
            "q25":          round(float(q25),             0),
            "q75":          round(float(q75),             0),
            "count":        int(len(prices)),
        }

    def calculate_sold_statistics(self, category_id=None) -> Dict:
        df = self.df.copy()
        if category_id:
            df = df[df["category_id"] == category_id]
        sold = df[df["sold_price"].notna()]
        if sold.empty:
            return {"median_sold": 0, "avg_discount_pct": 0, "avg_days": 0}
        return {
            "median_sold":      round(float(sold["sold_price"].median()), 0),
            "avg_discount_pct": round(float(sold["discount_rate"].mean() * 100), 1),
            "avg_days":         round(float(sold["days_to_sell"].median()), 0),
        }

    # ── 6. Нові методи для UI ─────────────────────────────────────────────────

    def market_speed_info(self, category_id: Optional[int] = None) -> Dict:
        """
        Індикатор швидкості ринку для UI.
        Повертає: середній час продажу, кількість активних, рівень конкуренції.
        """
        cat_id = category_id or 0
        cat_df = self.df[self.df["category_id"] == cat_id] if cat_id else self.df

        active_count = len(cat_df)
        avg_days     = float(cat_df["cat_days_median"].median()) if "cat_days_median" in cat_df.columns else 30

        # RESERVED як % від активних = швидкість ринку
        reserved_count = 0
        if not self.reserved_df.empty:
            reserved_count = len(
                self.reserved_df[self.reserved_df["category_id"] == cat_id]
            )

        if avg_days <= 3:
            speed_label = "🔥 Дуже швидкий"
        elif avg_days <= 7:
            speed_label = "⚡ Швидкий"
        elif avg_days <= 21:
            speed_label = "📊 Середній"
        else:
            speed_label = "🐢 Повільний"

        return {
            "avg_days_to_sell": round(avg_days, 0),
            "active_listings":  active_count,
            "reserved_count":   reserved_count,
            "speed_label":      speed_label,
        }

    def price_position(
        self,
        price: float,
        category_id: Optional[int] = None,
    ) -> Dict:
        """
        Де знаходиться ціна відносно ринку — для прогрес-бару в UI.
        Повертає percentile, лейбл і рекомендацію.
        """
        cat_id = category_id or 0
        cat_df = self.df[self.df["category_id"] == cat_id] if cat_id else self.df

        all_prices = pd.concat([
            cat_df["price"].dropna(),
            self.reserved_df["original_price"].dropna() if not self.reserved_df.empty else pd.Series(dtype=float),
        ])

        if all_prices.empty:
            return {"percentile": 50, "label": "середня", "recommendation": ""}

        pct = float((all_prices < price).mean() * 100)

        if pct < 20:
            label = "дуже низька"
            rec   = "Ціна нижча за 80% ринку. Можна підняти."
        elif pct < 40:
            label = "нижче середнього"
            rec   = "Хороша ціна для швидкого продажу."
        elif pct < 60:
            label = "середня ринкова"
            rec   = "Оптимальна ціна."
        elif pct < 80:
            label = "вище середнього"
            rec   = "Продаж може зайняти більше часу."
        else:
            label = "висока"
            rec   = "Ціна вища за 80% конкурентів. Рекомендуємо знизити."

        return {
            "percentile":     round(pct, 1),
            "label":          label,
            "recommendation": rec,
            "market_min":     round(float(all_prices.quantile(0.1)), 0),
            "market_median":  round(float(all_prices.median()), 0),
            "market_max":     round(float(all_prices.quantile(0.9)), 0),
        }

    def comparables_for_trust(
        self,
        query: str,
        category_id: Optional[int] = None,
        recommended_price: float = 0,
    ) -> Dict:
        """
        Блок довіри для UI: активні + продані + зарезервовані компаративи.
        Повертає структурований dict для фронтенду.
        """
        active_comps   = self.find_comparables(query, category_id, top_k=5)
        sold_comps     = self.find_fast_sold_comparables(query, category_id, top_k=3)
        reserved_comps = self.find_reserved_comparables(query, category_id, top_k=3)

        active_stats   = self.calculate_price_statistics(active_comps)
        sold_stats     = self.calculate_price_statistics(sold_comps)
        reserved_stats = self.calculate_price_statistics(reserved_comps)

        pos = self.price_position(recommended_price, category_id) if recommended_price else {}

        return {
            "active": {
                "items":  active_comps[["title", "price"]].to_dict("records"),
                "stats":  active_stats,
                "label":  "Активні оголошення",
            },
            "sold": {
                "items":  sold_comps[["title", "price"]].to_dict("records") if not sold_comps.empty else [],
                "stats":  sold_stats,
                "label":  "Реально продані (< 3 дні)",
            },
            "reserved": {
                "items":  reserved_comps[["title", "price"]].to_dict("records") if not reserved_comps.empty else [],
                "stats":  reserved_stats,
                "label":  "Зарезервовані (ціна спрацювала)",
            },
            "price_position": pos,
        }

    # ── 7. Ключові слова ──────────────────────────────────────────────────────

    def top_keywords(self, category_id: int, n: int = 15) -> List[str]:
        sub = self.df[self.df["category_id"] == category_id]["text"]
        if sub.empty:
            return []
        all_words = re.findall(r"[а-яёіїєa-z]{4,}", " ".join(sub))
        return [w for w, _ in Counter(all_words).most_common(n)]


# ── Допоміжні ─────────────────────────────────────────────────────────────────

def _cond_score(text: str) -> float:
    good = len(CONDITION_GOOD_RE.findall(text))
    bad  = len(CONDITION_BAD_RE.findall(text))
    if good == 0 and bad == 0:
        return 0.5
    return round(good / (good + bad + 1e-9), 3)


def _extract_num(text: str, pattern: str, default=0) -> float:
    m = re.search(pattern, text, re.I)
    return float(m.group(1)) if m else default