"""Live-market verification: same trust rules on real financial returns.

Fetches daily closes from Stooq (no key). Falls back to synthetic
GBM windows offline so experiments never break. Each window of log-returns
is one high-dim point (like one digit picture); t-SNE must keep neighbour
windows near without inventing island sizes / gaps.
"""
import io
import numpy as np

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ",
           "DIA", "IEF", "GLD", "BTC-USD"]
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def fetch_yahoo_daily(ticker, rng="2y", timeout=20):
    """Live closes via Yahoo v8 (no key). Verified live 2026-10-15, 200 OK."""
    import requests
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range={rng}"
    r = requests.get(url, timeout=timeout, headers=UA)
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]
    closes = [x for x in (j["indicators"]["adjclose"][0]["adjclose"]
                          if "adjclose" in j["indicators"]
                          else j["indicators"]["quote"][0]["close"]) if x]
    px = np.array(closes, dtype=float)
    if len(px) < 60:
        raise ValueError(f"too few rows for {ticker}: {len(px)}")
    return px


def fetch_stooq_daily(ticker, timeout=15):
    import requests
    url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
    r = requests.get(url, timeout=timeout, headers=UA)
    r.raise_for_status()
    if "<html" in r.text[:200].lower():
        raise ValueError("stooq bot-wall (HTML, not CSV)")
    lines = r.text.strip().splitlines()
    if len(lines) < 30:
        raise ValueError(f"too few rows for {ticker}")
    closes = []
    for ln in lines[1:]:
        parts = ln.split(",")
        closes.append(float(parts[4]))
    return np.array(closes)


def live_return_windows(n_win=16, win=30, seed=0):
    rets, labels, live = [], [], False
    try:
        for t in TICKERS:
            try:
                px = fetch_yahoo_daily(t)
            except Exception:
                px = fetch_stooq_daily(t.lower() + ".us")
            lr = np.diff(np.log(px))
            lr = lr[-(n_win * win):]
            for i in range(n_win):
                seg = lr[i * win:(i + 1) * win]
                if len(seg) == win:
                    rets.append((seg - seg.mean()) / (seg.std() + 1e-9))
                    labels.append(t)
        live = len(rets) >= 50
    except Exception:
        live = False
    if not live:
        rng = np.random.default_rng(seed)
        regimes = [(0.001, 0.01), (-0.001, 0.02), (0.0, 0.03)]
        rets, labels = [], []
        for k, (mu, sd) in enumerate(regimes):
            for _ in range(60):
                seg = rng.normal(mu, sd, win)
                rets.append((seg - seg.mean()) / (seg.std() + 1e-9))
                labels.append(f"regime-{k}")
    return np.array(rets), np.array(labels), live
