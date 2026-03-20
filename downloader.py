import os


def ensure_data(tickers, benchmark, data_dir="data", start="2000-01-01", end="2025-12-31"):
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance is not installed. Run: pip install yfinance")
        raise SystemExit(1)

    os.makedirs(data_dir, exist_ok=True)

    all_symbols = list(tickers) + [benchmark]
    missing = [s for s in all_symbols if not os.path.exists(os.path.join(data_dir, f"{s}.csv"))]

    if not missing:
        print("  All data files present — skipping download.")
        return

    for symbol in all_symbols:
        path = os.path.join(data_dir, f"{symbol}.csv")
        if os.path.exists(path):
            print(f"  [SKIP] {symbol} — already exists")
            continue

        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        df.to_csv(path)
        if len(df) > 0:
            date_start = df.index[0].strftime("%Y-%m-%d")
            date_end = df.index[-1].strftime("%Y-%m-%d")
            print(f"  [DL]   {symbol} — downloaded {len(df)} rows ({date_start} → {date_end})")
        else:
            print(f"  [DL]   {symbol} — downloaded 0 rows (no data returned)")
