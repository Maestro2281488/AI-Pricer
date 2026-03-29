"""
utils/price_analyzer.py

1. Регресія ціни  — Ridge(TF-IDF) + RandomForest(структурні фічі), log1p таргет
2. Стратегія      — percentile rank всередині категорії (замість RF класифікатора)
3. Компаративи    — keyword overlap
4. Статистика     — median, mean, IQR
"""

import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional, Dict, List

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_percentage_error
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings("ignore")


# ── Regex патерни ─────────────────────────────────────────────────────────────

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
    def __init__(self, df: pd.DataFrame, fast_sold: pd.DataFrame = None):
        self.df = df.copy()
        self.fast_sold = fast_sold
        self._build_features()
        self._train_regression()

    # ── Feature engineering ───────────────────────────────────────────────────

    def _build_features(self):
        d = self.df

        # Базові текстові метрики
        d["desc_len"]   = d["description"].fillna("").str.len()
        d["title_len"]  = d["title"].fillna("").str.len()
        d["word_count"] = d["description"].fillna("").str.split().str.len()

        # Бінарні сигнали з тексту
        d["has_brand"] = d["text"].apply(lambda x: int(bool(BRAND_RE.search(x))))
        d["cond_good"] = d["text"].apply(lambda x: int(bool(CONDITION_GOOD_RE.search(x))))
        d["cond_bad"]  = d["text"].apply(lambda x: int(bool(CONDITION_BAD_RE.search(x))))

        d["category_id"] = d["category_id"].fillna(0)

        # Медіана по категорії — в log-просторі (узгоджено з таргетом)
        cat_med = d.groupby("category_id")["price"].transform("median")
        d["cat_median"]     = cat_med.fillna(d["price"].median())
        d["log_cat_median"] = np.log1p(d["cat_median"])

        # Percentile rank ціни всередині категорії (0..1)
        # Використовується і як фіча, і для стратегій
        d["price_pct_rank"] = d.groupby("category_id")["price"].rank(pct=True)
        d["discount_rate"] = (
            (d["original_price"] - d["sold_price"]) / d["original_price"]
        ).clip(0, 1).fillna(0)
        cat_discount = d.groupby("category_id")["discount_rate"].transform("median")
        d["cat_discount_median"] = cat_discount.fillna(0)

        d["days_to_sell"] = (d["modified_at"] - d["created_at"]).dt.days.clip(0, 365)

        # Медіана по категорії
        cat_days = d.groupby("category_id")["days_to_sell"].transform("median")
        d["cat_days_median"] = cat_days.fillna(30)

    def _structural_features(self) -> List[str]:
        """Список структурних фіч — єдине місце, без дублювання."""
        return [
            "desc_len", "title_len", "has_brand",
            "cond_good", "cond_bad", "word_count",
            "category_id", "log_cat_median",
            "condition_score", "keyword_score",
        ]

    def _get_struct_matrix(self, df_slice: pd.DataFrame) -> pd.DataFrame:
        return df_slice[self._structural_features()].fillna(0)

    # ── 1. Регресія ───────────────────────────────────────────────────────────

    def _train_regression(self):
        valid = self.df[self.df["sold_price"].notna()].copy()
        y = np.log1p(valid["sold_price"].clip(upper=valid["sold_price"].quantile(0.99)))

        # ── Ridge + TF-IDF (добре з sparse, ловить назви брендів/моделей) ──
        self.tfidf = TfidfVectorizer(
            max_features=3000,
            min_df=5,            # ігноруємо слова що зустрічаються < 5 разів
            ngram_range=(1, 2),  # уніграми + біграми ("нові кросівки", "iphone pro")
            sublinear_tf=True,   # log(tf) замість tf — зменшує вплив частих слів
        )
        X_tfidf  = self.tfidf.fit_transform(valid["text"].fillna(""))
        X_struct = self._get_struct_matrix(valid)
        X_combined = hstack([csr_matrix(X_struct.values), X_tfidf])

        self.ridge = Ridge(alpha=10.0)
        self.ridge.fit(X_combined, y)

        # ── RandomForest тільки на структурних фічах ──
        # RF погано працює зі sparse матрицями, тому тримаємо окремо
        self.rf = RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        self.rf.fit(X_struct, y)

        # Метрика на train — інформаційно (не для хвастощів)
        y_pred_log = 0.6 * self.ridge.predict(X_combined) + 0.4 * self.rf.predict(X_struct)
        y_pred     = np.expm1(y_pred_log)
        y_real     = np.expm1(y)
        mape = mean_absolute_percentage_error(y_real, y_pred) * 100
        print(f"✅ Регресія навчена | Train MAPE: {mape:.1f}% (на log-таргеті)")

    def predict_price_regression(
        self,
        description: str,
        category_id: Optional[int] = None,
    ) -> float:
        cat_id  = category_id or 0
        cat_med = (
            self.df[self.df["category_id"] == cat_id]["price"].median()
            if cat_id in self.df["category_id"].values
            else self.df["price"].median()
        )

        text = description.lower()
        row = {
            "desc_len":        len(description),
            "title_len":       len(description.split()[:15]) * 6,
            "has_brand":       int(bool(BRAND_RE.search(text))),
            "cond_good":       int(bool(CONDITION_GOOD_RE.search(text))),
            "cond_bad":        int(bool(CONDITION_BAD_RE.search(text))),
            "word_count":      len(description.split()),
            "category_id":     cat_id,
            "log_cat_median":  np.log1p(cat_med),
            "condition_score": _cond_score(text),
            "keyword_score":   0.5,
        }
        X_struct   = pd.DataFrame([row])
        X_tfidf    = self.tfidf.transform([text])
        X_combined = hstack([csr_matrix(X_struct.values), X_tfidf])

        ridge_log = float(self.ridge.predict(X_combined)[0])
        rf_log    = float(self.rf.predict(X_struct)[0])

        # Ансамбль: Ridge краще знає "що це за товар", RF — "скільки коштують схожі"
        pred_log = 0.6 * ridge_log + 0.4 * rf_log
        pred     = float(np.expm1(pred_log))
        return max(50.0, round(pred / 50) * 50)

    # ── 2. Стратегія продажу ──────────────────────────────────────────────────
    # Замість percentile rank на original_price - рахуємо на реальних угодах
    def classify_strategy(self, recommended_price, category_id=None):
        cat_id = category_id or 0
        
        # Беремо sold_price якщо є
        sold = self.df[
            (self.df["category_id"] == cat_id) & 
            self.df["sold_price"].notna()
        ]["sold_price"]
        
        prices = sold if len(sold) >= 10 else self.df[self.df["category_id"] == cat_id]["price"]
        
        if prices.empty:
            pct = 0.5
        else:
            pct = float((prices < recommended_price).mean())

        # Медіана дисконту по категорії
        cat_discount = self.df[
            (self.df["category_id"] == cat_id) & 
            self.df["discount_rate"].notna()
        ]["discount_rate"].median()
        
        fast_discount = max(0.10, cat_discount * 1.5)  # FAST = типовий дисконт * 1.5
        max_premium   = 0.25 if pct > 0.65 else 0.20

        return {
            "FAST":       max(50, int(round(recommended_price * (1 - fast_discount) / 50) * 50)),
            "BALANCED":   max(50, int(round(recommended_price / 50) * 50)),
            "MAX_PROFIT": max(50, int(round(recommended_price * (1 + max_premium) / 50) * 50)),
            "_label":     "high" if pct > 0.65 else "medium" if pct > 0.35 else "low",
            "_pct_rank":  round(pct, 2),
        }

    # ── 3. Компаративи ────────────────────────────────────────────────────────

    def find_comparables(
        self,
        query: str,
        category_id: Optional[int] = None,
        top_k: int = 8,
        source_df: Optional[pd.DataFrame] = None,  # ← новий параметр
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

    # ── 4. Статистика ─────────────────────────────────────────────────────────

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

    # ── 5. Ключові слова категорії ────────────────────────────────────────────

    def top_keywords(self, category_id: int, n: int = 15) -> List[str]:
        """Топ-N слів категорії для контексту промпту."""
        sub = self.df[self.df["category_id"] == category_id]["text"]
        if sub.empty:
            return []
        all_words = re.findall(r"[а-яёіїєa-z]{4,}", " ".join(sub))
        return [w for w, _ in Counter(all_words).most_common(n)]


    # Пошук товарів з оптимальною ціною (продалися швидко)

    def find_fast_sold_comparables(self, query: str, category_id=None, top_k=5):
        """Компаративи з реально проданих товарів — найнадійніша ціна."""
        if self.fast_sold is None or self.fast_sold.empty:
            return pd.DataFrame()
        return self.find_comparables(
            query, category_id, top_k, source_df=self.fast_sold
        )
    
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

# ── Допоміжна функція ─────────────────────────────────────────────────────────

def _cond_score(text: str) -> float:
    """0.0 = поганий стан, 1.0 = новий, 0.5 = невідомо."""
    good = len(CONDITION_GOOD_RE.findall(text))
    bad  = len(CONDITION_BAD_RE.findall(text))
    if good == 0 and bad == 0:
        return 0.5
    return round(good / (good + bad + 1e-9), 3)