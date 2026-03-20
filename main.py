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
from downloader import ensure_data
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
)


def main():
    start_time = time.time()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_data(cfg.UNIVERSE, cfg.BENCHMARK, data_dir="data",
                start=cfg.START_DATE, end=cfg.END_DATE)

    # ── Banner ─────────────────────────────────────────────────────────────────
    print_header(
        "AI STOCK TRADER — REAL DATA",
        subtitle=f"Ensemble ×{cfg.ENSEMBLE_MODELS} MLP | {len(cfg.UNIVERSE)} real stocks | vs S&P 500"
    )

    # ── 1. LOAD REAL DATA ─────────────────────────────────────────────────────
    print_section("1. LOADING REAL STOCK DATA")
    print(f"  Loading OHLCV from data/ (downloaded via yfinance) …")
    data = load_real_data(cfg.UNIVERSE, data_dir="data")
    prices = data["Close"]
    benchmark = data["Benchmark"]

    print_metric("Tickers", ", ".join(cfg.UNIVERSE))
    print_metric("Benchmark", "SPY (S&P 500 ETF)")
    print_metric("Date range", f"{prices.index[0].date()} → {prices.index[-1].date()}")
    print_metric("Trading days", len(prices))
    print_metric("Data source", "Yahoo Finance (real historical)")

    # Show per-stock returns
    total_rets = (prices.iloc[-1] / prices.iloc[0] - 1)
    best = total_rets.idxmax()
    worst = total_rets.idxmin()
    print_metric(f"Best performer", f"{best} ({total_rets[best]:+.0%})")
    print_metric(f"Worst performer", f"{worst} ({total_rets[worst]:+.0%})")
    spy_ret = benchmark.iloc[-1] / benchmark.iloc[0] - 1
    print_metric(f"SPY total return", f"{spy_ret:+.0%}")

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
        X, y, dates_seq = build_sequences(normed, close_raw, cfg.LOOKBACK_WINDOW, fwd_days=5)
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
    print_metric("Target", "Forward 5-day raw price return")

    # Split by DATE (temporal — no future leakage)
    train_mask = np.array([d <= split_date for d in all_dates])
    test_mask = ~train_mask

    X_train_full = X_all[train_mask]
    y_train_full = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]
    test_dates_list = [d for d, m in zip(all_dates, test_mask) if m]
    test_tickers_list = [t for t, m in zip(all_tickers_list, test_mask) if m]

    # Train / val split (within training period)
    val_split = int(len(X_train_full) * cfg.VAL_SPLIT_RATIO)
    X_tr, X_val = X_train_full[:val_split], X_train_full[val_split:]
    y_tr, y_val = y_train_full[:val_split], y_train_full[val_split:]

    print_metric("Train samples", f"{len(X_tr):,}")
    print_metric("Validation samples", f"{len(X_val):,}")
    print_metric("Test samples", f"{len(X_test):,}")

    # ── 5. TRAINING ──────────────────────────────────────────────────────────
    print_section("5. ENSEMBLE TRAINING")
    seed_offsets = [0, 17, 41, 73, 97, 131, 167, 199]  # prime offsets for max diversity
    models = []
    for i in range(cfg.ENSEMBLE_MODELS):
        np.random.seed(42 + seed_offsets[i])
        n = len(X_tr)
        idx = np.random.choice(n, size=int(n * cfg.BOOTSTRAP_RATIO), replace=True)
        X_boot, y_boot = X_tr[idx], y_tr[idx]
        m = train_model(X_boot, y_boot, X_val, y_val, i, cfg)
        models.append(m)

    print(f"\n  ✓  Ensemble of {cfg.ENSEMBLE_MODELS} models trained on real stock data.")

    # ── 6. PREDICTION ────────────────────────────────────────────────────────
    print_section("6. GENERATING PREDICTIONS")

    # ── Step 1: Validate model on val set to calibrate signal direction ──
    val_preds = np.zeros(len(X_val))
    for m in models:
        val_preds += m.predict(X_val)
    val_preds /= len(models)
    val_corr = np.corrcoef(val_preds, y_val)[0, 1]
    signal_flip = -1.0 if val_corr < 0 else 1.0
    print_metric("Validation correlation", f"{val_corr:.4f}")
    print_metric("Signal calibration", "FLIPPED (val corr < 0)" if signal_flip < 0 else "NORMAL")

    # ── Step 2: Generate test predictions with calibrated direction ──
    preds = np.zeros(len(X_test))
    for m in models:
        preds += m.predict(X_test)
    preds /= len(models)
    preds *= signal_flip  # flip if validation showed negative correlation

    # Signal quality (calibrated model)
    corr = np.corrcoef(preds, y_test)[0, 1]
    dir_acc = np.mean(np.sign(preds) == np.sign(y_test))
    print_metric("Test signal-target correlation", f"{corr:.4f}")
    print_metric("Test direction accuracy", f"{dir_acc:.2%}")

    # Z-score standardize
    pred_mean = preds.mean()
    pred_std = max(preds.std(), 1e-10)
    preds_z = np.clip((preds - pred_mean) / pred_std, -3.0, 3.0)

    # Map back to (date, ticker)
    pred_df = pd.DataFrame({
        "date": test_dates_list,
        "ticker": test_tickers_list,
        "pred": preds_z,
    })
    model_pivot = pred_df.pivot_table(index="date", columns="ticker", values="pred", aggfunc="mean")
    model_pivot = model_pivot.reindex(columns=cfg.UNIVERSE).fillna(0.0)

    # ── Momentum overlay: proven alpha factor ──
    print_section("6b. MOMENTUM OVERLAY")
    mom_lookback = getattr(cfg, 'MOMENTUM_LOOKBACK', 60)
    mom_weight = getattr(cfg, 'MOMENTUM_WEIGHT', 0.65)
    mdl_weight = getattr(cfg, 'MODEL_WEIGHT', 0.35)

    # Compute trailing momentum for each stock at each date
    mom_signals = prices.pct_change(mom_lookback)
    # Cross-sectional rank: on each date, rank stocks by momentum [0, 1]
    mom_ranked = mom_signals.rank(axis=1, pct=True)
    # Center to [-0.5, 0.5] so it's symmetric
    mom_centered = mom_ranked - 0.5
    # Align to test dates
    mom_pivot = mom_centered.reindex(model_pivot.index).fillna(0.0)
    mom_pivot = mom_pivot.reindex(columns=cfg.UNIVERSE).fillna(0.0)

    # Also cross-sectionally rank the model signals per date
    model_ranked = model_pivot.rank(axis=1, pct=True) - 0.5

    # Blend: momentum + calibrated model
    pred_pivot = mom_weight * mom_pivot + mdl_weight * model_ranked

    print_metric("Momentum weight", f"{mom_weight:.0%}")
    print_metric("Model weight", f"{mdl_weight:.0%}")
    print_metric("Momentum lookback", f"{mom_lookback} days")
    print_metric("Blended signal range",
                 f"[{pred_pivot.values.min():.3f}, {pred_pivot.values.max():.3f}]")
    print_metric("Prediction days", len(pred_pivot))

    # ── 7. BACKTESTING ───────────────────────────────────────────────────────
    print_section("7. BACKTESTING ON REAL PRICES")
    test_prices = prices.reindex(pred_pivot.index).ffill().bfill()
    bench = benchmark.reindex(pred_pivot.index).ffill().bfill()
    portfolio, trades = run_backtest(pred_pivot, test_prices, bench, cfg)

    metrics = compute_metrics(portfolio, trades)

    # ── 8. RESULTS ───────────────────────────────────────────────────────────
    print_section("8. PERFORMANCE SUMMARY — AI vs S&P 500")

    table = [
        ["Metric", "AI Strategy", "S&P 500 (SPY)"],
        ["Total Return", f"{metrics['strategy_return']:.2%}", f"{metrics['benchmark_return']:.2%}"],
        ["CAGR", f"{metrics['strategy_cagr']:.2%}", f"{metrics['benchmark_cagr']:.2%}"],
        ["Sharpe Ratio", f"{metrics['strategy_sharpe']:.3f}", f"{metrics['benchmark_sharpe']:.3f}"],
        ["Sortino Ratio", f"{metrics['strategy_sortino']:.3f}", f"{metrics['benchmark_sortino']:.3f}"],
        ["Max Drawdown", f"{metrics['strategy_max_dd']:.2%}", f"{metrics['benchmark_max_dd']:.2%}"],
        ["Annualised Vol", f"{metrics['strategy_vol']:.2%}", f"{metrics['benchmark_vol']:.2%}"],
        ["Calmar Ratio", f"{metrics['strategy_calmar']:.3f}", f"{metrics['benchmark_calmar']:.3f}"],
        ["Win Rate", f"{metrics['win_rate']:.1%}", "—"],
        ["Profit Factor", f"{metrics['profit_factor']:.2f}", "—"],
        ["Total Trades", f"{metrics['total_trades']:,}", "—"],
        ["Avg Trade Ret", f"{metrics['avg_trade_return']:.4%}", "—"],
    ]
    print_table(table)

    beat = metrics["strategy_return"] > metrics["benchmark_return"]
    alpha = metrics["strategy_return"] - metrics["benchmark_return"]
    verdict = colour("BEAT", "g") if beat else colour("UNDERPERFORMED", "r")
    clr = "g" if beat else "r"

    print(f"\n  {'━' * 62}")
    print(f"  AI Strategy {verdict} the S&P 500 by {colour(f'{alpha:+.2%}', clr)}")
    print(f"  {'━' * 62}")

    # ── 8a. WINNING / LOSING STREAKS ─────────────────────────────────────────
    print_section("8a. WIN/LOSS STREAKS")
    import itertools
    daily_rets = metrics['strat_daily']
    streaks_binary = (daily_rets > 0).astype(int)
    max_win_streak = max(
        (sum(1 for _ in g) for k, g in itertools.groupby(streaks_binary) if k == 1),
        default=0,
    )
    max_loss_streak = max(
        (sum(1 for _ in g) for k, g in itertools.groupby(streaks_binary) if k == 0),
        default=0,
    )
    print_metric("Max winning streak", f"{max_win_streak} days")
    print_metric("Max losing streak", f"{max_loss_streak} days")
    win_pct = (daily_rets > 0).mean()
    print_metric("Positive days", f"{(daily_rets > 0).sum()} / {len(daily_rets)} ({win_pct:.1%})")
    avg_win = daily_rets[daily_rets > 0].mean() * 100
    avg_loss = daily_rets[daily_rets < 0].mean() * 100
    print_metric("Avg winning day", f"+{avg_win:.3f}%")
    print_metric("Avg losing day", f"{avg_loss:.3f}%")
    print_metric("Win/Loss magnitude", f"{abs(avg_win / avg_loss):.2f}x")

    # ── 8b. ALPHA DECOMPOSITION ──────────────────────────────────────────────
    print_section("8b. ALPHA DECOMPOSITION")
    long_trades = trades[trades["side"].isin(["BUY", "SELL"])]
    short_trades = trades[trades["side"].isin(["SHORT", "COVER", "STOP", "TAKE_PROFIT", "DELEVERAGE"])]
    total_long = len(long_trades)
    total_short = len(short_trades)
    short_long_ratio = total_short / max(total_long, 1)
    print_metric("Total long trades", f"{total_long:,}")
    print_metric("Total short trades", f"{total_short:,}")
    print_metric("Short/Long ratio", f"{short_long_ratio:.1%}")
    tracking_error = metrics.get("tracking_error", 0.01)
    info_ratio = alpha / max(tracking_error, 0.001)
    print_metric("Tracking error", f"{tracking_error:.2%}")
    print_metric("Information ratio", f"{info_ratio:.3f}")
    print_metric("Alpha (annualised)", f"{alpha:+.2%}")

    # ── 8c. RISK-ADJUSTED PERFORMANCE ────────────────────────────────────────
    print_section("8c. RISK-ADJUSTED PERFORMANCE")
    risk_rows = [
        ["Metric", "Value", "Rating"],
        [
            "Sharpe Ratio",
            f"{metrics['strategy_sharpe']:.3f}",
            colour("Excellent", "g") if metrics['strategy_sharpe'] > 1.5
            else colour("Good", "y") if metrics['strategy_sharpe'] > 1.0
            else colour("Fair", "r"),
        ],
        [
            "Sortino Ratio",
            f"{metrics['strategy_sortino']:.3f}",
            colour("Excellent", "g") if metrics['strategy_sortino'] > 2.0
            else colour("Good", "y") if metrics['strategy_sortino'] > 1.5
            else colour("Fair", "r"),
        ],
        [
            "Max Drawdown",
            f"{metrics['strategy_max_dd']:.2%}",
            colour("Low Risk", "g") if abs(metrics['strategy_max_dd']) < 0.10
            else colour("Moderate", "y") if abs(metrics['strategy_max_dd']) < 0.20
            else colour("High", "r"),
        ],
        [
            "Calmar Ratio",
            f"{metrics['strategy_calmar']:.3f}",
            colour("Excellent", "g") if metrics['strategy_calmar'] > 2.0
            else colour("Good", "y") if metrics['strategy_calmar'] > 1.0
            else colour("Fair", "r"),
        ],
        [
            "Volatility",
            f"{metrics['strategy_vol']:.2%}",
            colour("Low", "g") if metrics['strategy_vol'] < 0.15
            else colour("Moderate", "y") if metrics['strategy_vol'] < 0.25
            else colour("High", "r"),
        ],
    ]
    print_table(risk_rows)

    # ── 9. REPORTS & CHARTS ──────────────────────────────────────────────────
    print_section("9. GENERATING OUTPUTS")
    generate_report(metrics, trades, portfolio, cfg.REPORT_PATH, tickers=cfg.UNIVERSE, cfg_mod=cfg)
    print(f"  ✓  Report  → {cfg.REPORT_PATH}")

    plot_charts(portfolio, trades, metrics, cfg.CHART_PATH, cfg)
    print(f"  ✓  Charts  → {cfg.CHART_PATH}")

    elapsed = time.time() - start_time
    print(f"\n  ⏱  Total runtime: {elapsed:.1f}s\n")
    print_header("COMPLETE", subtitle="Real stock backtest finished — see outputs/")


if __name__ == "__main__":
    main()
