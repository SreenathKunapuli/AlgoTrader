"""Sample headlines for the LLM-sentiment distillation pilot.

Pulls pages of Benzinga headlines via Alpaca News across 2022-2026, keeps
articles tagged with <=3 symbols (sentiment is per-article; a 20-ticker
market-recap headline has no per-stock polarity), dedupes, saves CSV.
Teacher labels are produced by Claude in-session; the student is TF-IDF +
logistic regression trained on those labels (see xsec/sentiment.py).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from alpaca.data.historical.news import NewsClient  # noqa: E402
from alpaca.data.requests import NewsRequest  # noqa: E402

WINDOWS = [(datetime(y, m, 1), datetime(y, m, 21))
           for y in (2022, 2023, 2024, 2025, 2026)
           for m in (2, 5, 8, 11) if not (y == 2026 and m > 5)]


def main() -> None:
    c = NewsClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    rows = []
    for start, end in WINDOWS:
        page = c.get_news(NewsRequest(start=start, end=end, limit=50))
        for a in page.data["news"]:
            if a.symbols and len(a.symbols) <= 3 and a.headline:
                rows.append({"created_at": a.created_at, "headline": a.headline.strip(),
                             "symbols": ",".join(a.symbols)})
        print(f"{start.date()}: total {len(rows)}", flush=True)
    df = pd.DataFrame(rows).drop_duplicates(subset="headline").reset_index(drop=True)
    df.index.name = "id"
    out = ROOT / "data/news_sample.csv"
    df.to_csv(out)
    print(f"saved {len(df)} unique headlines -> {out}")


if __name__ == "__main__":
    main()
