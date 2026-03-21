#!/usr/bin/env python3
"""
AI Stock Trader — Main Entry Point
====================================
Loads REAL historical stock data (AAPL, MSFT, NVDA, etc.), trains an
ensemble of neural networks on 40+ technical features, and backtests
a portfolio against the actual S&P 500 (SPY).

Bias-free pipeline:
  - Features normalised with TRAINING data stats only (no look-ahead)
  - Target = raw forward 5-day price return (not z-scored differences)
  - Proper temporal train/test split

Outputs:
  • Rich terminal summary with formatted tables
  • Detailed text report   → outputs/trading_report.txt
  • 6-panel chart image    → outputs/trading_charts.png

Usage:
    python main.py
"""

import sys
import time
import numpy as np
import pandas as pd

import config as cfg
from utils import (
    load_real_data,
    add_features,
    fit_normalisation,
    apply_normalisation,
    build_sequences,
    train_model,
    run_backtest,
    compute_metrics,
    generate_report,
    plot_charts,
    print_header,
    print_section,
    print_metric,
    print_table,
    pbar,
    colour,
    colour_cmp,
)


def main():
    start_time = time.time()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Banner ─────────────────────────────────────────────────────────────────
    print_header(
        "AI STOCK TRADER — REAL DATA",
        subtitle=f"Ensemble ×{cfg.ENSEMBLE_MODELS} MLP | {len(cfg.UNIVERSE)} real stocks | vs S&P 500"
    )

    # ── 1. LOAD REAL DATA ─────────────────────────────────────────────────────
    print_section("1. LOADING REAL STOCK DATA")
    print(f"  Loading OHLCV from data/ (auto-downloads missing files via yfinance) …")
    data = load_real_data(
        cfg.UNIVERSE,
        data_dir=cfg.DATA_DIR,
        benchmark=cfg.BENCHMARK,
        start_date=cfg.START_DATE,
        end_date=cfg.END_DATE,
        auto_download=True,
    )
    prices = data["Close"]
    benchmark = data["Benchmark"]

    print_metric("Tickers", ", ".join(cfg.UNIVERSE))
    print_metric("Benchmark", f"{cfg.BENCHMARK} (S&P 500 ETF)")
    print_metric("Date range", f"{prices.index[0].date()} → {prices.index[-1].date()}")
    print_metric("Trading days", len(prices))
    print_metric("Data source", "Yahoo Finance (real historical; auto-download enabled)")

    # Show per-stock returns
    total_rets = (prices.iloc[-1] / prices.iloc[0] - 1)
    best = total_rets.idxmax()
    worst = total_rets.idxmin()
    print_metric("Best performer", f"{best} ({total_rets[best]:+.0%})")
    print_metric("Worst performer", f"{worst} ({total_rets[worst]:+.0%})")
    spy_ret = benchmark.iloc[-1] / benchmark.iloc[0] - 1
    print_metric("Benchmark total return", f"{spy_ret:+.0%}")

    # ── 2. FEATURES (RAW, un-normalised) ──────────────────────────────────────
    print_section("2. FEATURE ENGINEERING")
    raw_feature_frames = {}
    for ticker in cfg.UNIVERSE:
        feat = add_features(
            data["Close"][ticker], data["High"][ticker],
            data["Low"][ticker], data["Volume"][ticker],
        )
        raw_feature_frames[ticker] = feat

    n_feat = raw_feature_frames[cfg.UNIVERSE[0]].shape[1]
    print_metric("Features per ticker", n_feat)
    print_metric("Feature categories", "Returns, MAs, MACD, RSI, BB, ATR, Stoch, CCI, OBV, MFI, Vol")
    print_metric("Normalisation", "Train-only Z-score (bias-free)")

    # ── 3. TEMPORAL SPLIT + NORMALISATION ─────────────────────────────────────
    print_section("3. TEMPORAL SPLIT & NORMALISATION")

    # Find common date range across all tickers
    common_dates = raw_feature_frames[cfg.UNIVERSE[0]].index
    for t in cfg.UNIVERSE[1:]:
        common_dates = common_dates.intersection(raw_feature_frames[t].index)
    common_dates = common_dates.sort_values()

    # Split dates: 70% train, 30% test (temporal, no shuffling)
    split_idx = int(len(common_dates) * (1 - cfg.TEST_SPLIT))
    train_dates = common_dates[:split_idx]
    test_dates = common_dates[split_idx:]
    split_date = train_dates[-1]

    print_metric("Train period", f"{train_dates[0].date()} → {train_dates[-1].date()}")
    print_metric("Test period", f"{test_dates[0].date()} → {test_dates[-1].date()}")
    print_metric("Train days", len(train_dates))
    print_metric("Test days", len(test_dates))

    # Fit normalisation on TRAINING data only, then apply to both splits
    norm_feature_frames = {}
    norm_stats = {}
    for ticker in cfg.UNIVERSE:
        raw = raw_feature_frames[ticker]
        # Only use training period rows for computing mu/sigma
        train_rows = raw.loc[raw.index.isin(train_dates)]
        mu, sigma = fit_normalisation(train_rows)
        norm_stats[ticker] = (mu, sigma)
        # Apply same stats to the FULL feature frame
        normed = apply_normalisation(raw, mu, sigma)
        norm_feature_frames[ticker] = normed

    print_metric("Normalisation fit on", "Training data ONLY (no look-ahead)")

    # ── 4. BUILD SEQUENCES ───────────────────────────────────────────────────
    print_section("4. BUILDING SEQUENCES")
    all_X, all_y, all_dates, all_tickers_list = [], [], [], []

    for ticker in cfg.UNIVERSE:
        normed = norm_feature_frames[ticker]
        close_raw = data["Close"][ticker]
        X, y, dates_seq = build_sequences(normed, close_raw, cfg.LOOKBACK_WINDOW, fwd_days=cfg.FORWARD_DAYS)
        all_X.append(X)
        all_y.append(y)
        all_dates.extend(dates_seq)
        all_tickers_list.extend([ticker] * len(y))

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    input_dim = X_all.shape[1]

    # NaN/Inf safety
    nan_mask = np.isnan(X_all).any(axis=1) | np.isnan(y_all) | np.isinf(X_all).any(axis=1) | np.isinf(y_all)
    if nan_mask.any():
        print(f"  ⚠  Removing {nan_mask.sum()} samples with NaN/Inf")
        X_all = X_all[~nan_mask]
        y_all = y_all[~nan_mask]
        all_dates = [d for d, m in zip(all_dates, nan_mask) if not m]
        all_tickers_list = [t for t, m in zip(all_tickers_list, nan_mask) if not m]

    print_metric("Total samples", f"{X_all.shape[0]:,}")
    print_metric("Input dimension", f"{input_dim} (window={cfg.LOOKBACK_WINDOW} × features={n_feat})")
    print_metric("Target", f"Forward {cfg.FORWARD_DAYS}-day raw price return")

    # Split by DATE (temporal — no future leakage)
    # Safety buffer: each training target uses cfg.FORWARD_DAYS of future prices.
    # Samples within FORWARD_DAYS of the nominal split date would have targets
    # that land in the test period, creating lookahead bias. We exclude them
    # from training by moving the effective split boundary back by FORWARD_DAYS.
    _safe_idx = max(0, split_idx - 1 - cfg.FORWARD_DAYS)
    safe_split_date = common_dates[_safe_idx]
    train_mask = np.array([d <= safe_split_date for d in all_dates])
    test_mask = ~train_mask

    X_train_full = X_all[train_mask]
    y_train_full = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]
    test_dates_list = [d for d, m in zip(all_dates, test_mask) if m]
    test_tickers_list = [t for t, m in zip(all_tickers_list, test_mask) if m]

    # Train / val split (within training period)
    val_split = int(len(X_train_full) * 0.82)
    X_tr, X_val = X_train_full[:val_split], X_train_full[val_split:]
    y_tr, y_val = y_train_full[:val_split], y_train_full[val_split:]

    print_metric("Train samples", f"{len(X_tr):,}")
    print_metric("Validation samples", f"{len(X_val):,}")
    print_metric("Test samples", f"{len(X_test):,}")

    # ── 5. TRAINING ──────────────────────────────────────────────────────────
    print_section("5. ENSEMBLE TRAINING")
    # Use an isolated RandomState for bootstrap sampling to avoid polluting
    # the global numpy RNG (NumpyMLP uses its own internal RandomState).
    _boot_rng = np.random.RandomState(42)
    models = []
    for i in range(cfg.ENSEMBLE_MODELS):
        n = len(X_tr)
        idx = _boot_rng.choice(n, size=int(n * cfg.BOOTSTRAP_RATIO), replace=True)
        X_boot, y_boot = X_tr[idx], y_tr[idx]
        m = train_model(X_boot, y_boot, X_val, y_val, i, cfg)
        models.append(m)

    print(f"\n  {colour('✓', 'g')}  Ensemble of {colour(str(cfg.ENSEMBLE_MODELS), 'c')} models trained on real historical data.")

    # ── 6. PREDICTION ────────────────────────────────────────────────────────
    print_section("6. GENERATING PREDICTIONS")
    preds = np.zeros(len(X_test))
    for m in models:
        preds += m.predict(X_test)
    preds /= len(models)

    # Map back to (date, ticker)
    pred_df = pd.DataFrame({
        "date": test_dates_list,
        "ticker": test_tickers_list,
        "pred": preds,
    })
    pred_pivot = pred_df.pivot_table(index="date", columns="ticker", values="pred", aggfunc="mean")
    pred_pivot = pred_pivot.reindex(columns=cfg.UNIVERSE).fillna(0.0)

    # Signal quality
    corr = np.corrcoef(preds, y_test)[0, 1] if preds.std() > 0 else 0.0
    # Use strict inequality to avoid sign(0)==sign(0) inflating accuracy
    dir_acc = np.mean((preds > 0) == (y_test > 0))

    print_metric("Prediction days", len(pred_pivot))
    print_metric("Signal-target correlation", f"{corr:.4f}")
    print_metric("Direction accuracy", f"{dir_acc:.2%}")
    print_metric("Mean signal", f"{preds.mean():.6f}")
    print_metric("Signal std", f"{preds.std():.6f}")

    # ── 7. BACKTESTING ───────────────────────────────────────────────────────
    print_section("7. BACKTESTING ON REAL PRICES")
    # Forward-fill only — bfill would use future prices to fill leading gaps.
    test_prices = prices.reindex(pred_pivot.index).ffill()
    bench = benchmark.reindex(pred_pivot.index).ffill()
    portfolio, trades = run_backtest(pred_pivot, test_prices, bench, cfg)

    metrics = compute_metrics(portfolio, trades, rf=cfg.RISK_FREE_RATE)

    # ── 8. RESULTS ───────────────────────────────────────────────────────────
    print_section("8. PERFORMANCE SUMMARY — AI vs S&P 500")

    m = metrics
    table = [
        ["Metric",             "AI Strategy",                                                          "S&P 500 (SPY)"],
        ["Total Return",       colour_cmp(m['strategy_return'],  m['benchmark_return'],  ".2%"),       f"{m['benchmark_return']:.2%}"],
        ["CAGR",               colour_cmp(m['strategy_cagr'],    m['benchmark_cagr'],    ".2%"),       f"{m['benchmark_cagr']:.2%}"],
        ["Sharpe Ratio",       colour_cmp(m['strategy_sharpe'],  m['benchmark_sharpe'],  ".3f"),       f"{m['benchmark_sharpe']:.3f}"],
        ["Sortino Ratio",      colour_cmp(m['strategy_sortino'], m['benchmark_sortino'], ".3f"),       f"{m['benchmark_sortino']:.3f}"],
        ["Max Drawdown",       colour_cmp(m['strategy_max_dd'],  m['benchmark_max_dd'],  ".2%"),       f"{m['benchmark_max_dd']:.2%}"],
        ["Annualised Vol",     colour_cmp(m['strategy_vol'],     m['benchmark_vol'],     ".2%", False),f"{m['benchmark_vol']:.2%}"],
        ["Calmar Ratio",       colour_cmp(m['strategy_calmar'],  m['benchmark_calmar'],  ".3f"),       f"{m['benchmark_calmar']:.3f}"],
        ["Positive Days",      f"{m['win_rate']:.1%}",           "—"],
        ["Daily Prof Factor",  f"{m['profit_factor']:.2f}",      "—"],
        ["Total Trades",       f"{m['total_trades']:,}",          "—"],
        ["Avg Trade Size",     f"{m['avg_trade_size']:.4%}",      "—"],
    ]
    print_table(table)

    beat = metrics["strategy_return"] > metrics["benchmark_return"]
    alpha = metrics["strategy_return"] - metrics["benchmark_return"]
    verdict = colour("BEAT", "g") if beat else colour("UNDERPERFORMED", "r")
    clr = "g" if beat else "r"
    w = 66

    print(f"\n  {colour('━' * w, clr)}")
    print(f"  AI Strategy {verdict} the S&P 500 by {colour(f'{alpha:+.2%}', clr)}")
    detail = (f"  Strategy {metrics['strategy_return']:.2%}  ·  "
              f"Benchmark {metrics['benchmark_return']:.2%}  ·  "
              f"Sharpe {metrics['strategy_sharpe']:.3f}  ·  "
              f"Sortino {metrics['strategy_sortino']:.3f}")
    print(colour(detail, "D"))
    print(f"  {colour('━' * w, clr)}")

    # ── 9. REPORTS & CHARTS ──────────────────────────────────────────────────
    print_section("9. GENERATING OUTPUTS")
    generate_report(metrics, trades, portfolio, cfg.REPORT_PATH, tickers=cfg.UNIVERSE)
    print(f"  ✓  Report  → {cfg.REPORT_PATH}")

    plot_charts(portfolio, trades, metrics, cfg.CHART_PATH, cfg)
    print(f"  ✓  Charts  → {cfg.CHART_PATH}")

    elapsed = time.time() - start_time
    print(f"\n  ⏱  Total runtime: {elapsed:.1f}s\n")
    print_header("COMPLETE", subtitle="Real stock backtest finished — see outputs/")


if __name__ == "__main__":
    main()
