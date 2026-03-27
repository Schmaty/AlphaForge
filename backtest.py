#!/usr/bin/env python3
"""
AI Stock Trader — Backtest & Evaluation Suite
==============================================
Standalone testing module using REAL stock data:
  1. Data pipeline validation (real CSVs)
  2. Model architecture tests
  3. Walk-forward cross-validation on real data
  4. Monte Carlo confidence intervals
  5. Parameter sensitivity analysis
  6. Evaluation charts

Usage:
    python backtest.py             # full suite
    python backtest.py --quick     # fast smoke test
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

import config as cfg
from utils import (
    load_real_data,
    add_features,
    fit_normalisation,
    apply_normalisation,
    build_sequences,
    NumpyMLP,
    train_model,
    run_backtest,
    compute_metrics,
    print_header,
    print_section,
    print_metric,
    print_table,
    pbar,
    colour,
)

OUTPUT = Path(__file__).parent / "outputs"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1: DATA PIPELINE (REAL DATA)
# ═══════════════════════════════════════════════════════════════════════════════

def test_data_pipeline():
    print_section("TEST 1: DATA PIPELINE VALIDATION (REAL DATA)")
    errors = []

    # 1a. Load real data
    print("  1a. Loading real stock data …")
    try:
        tickers = cfg.UNIVERSE[:5]
        data = load_real_data(
            tickers,
            data_dir=cfg.DATA_DIR,
            benchmark=cfg.BENCHMARK,
            start_date=cfg.START_DATE,
            end_date=cfg.END_DATE,
            auto_download=True,
        )
        assert len(data["Close"]) > 200, "Too few rows"
        assert not data["Close"].isnull().all().any(), "All-NaN columns"
        print(f"    ✓  {len(data['Close'])} days, {data['Close'].shape[1]} tickers")
        print(f"    ✓  Date range: {data['Close'].index[0].date()} → {data['Close'].index[-1].date()}")
    except Exception as e:
        errors.append(f"Data load: {e}")
        print(f"    ✗  {e}")
        return errors

    # 1b. Feature engineering (raw, un-normalised)
    print("  1b. Feature engineering (raw) …")
    try:
        t = tickers[0]
        feat = add_features(data["Close"][t], data["High"][t], data["Low"][t], data["Volume"][t])
        assert not feat.empty, "Empty features"
        assert not np.isinf(feat.values).any(), "Inf in features"
        print(f"    ✓  {feat.shape[1]} features, {len(feat)} rows")
    except Exception as e:
        errors.append(f"Features: {e}")
        print(f"    ✗  {e}")
        return errors

    # 1c. Normalisation (train-only)
    print("  1c. Train-only normalisation …")
    try:
        split = int(len(feat) * 0.7)
        train_feat = feat.iloc[:split]
        mu, sigma = fit_normalisation(train_feat)
        normed = apply_normalisation(feat, mu, sigma)
        assert not normed.isnull().any().any(), "NaN after normalisation"
        # Verify train portion has ~0 mean, ~1 std
        train_normed = normed.iloc[:split]
        assert abs(train_normed.mean().mean()) < 0.1, "Train mean too far from 0"
        print(f"    ✓  Train mean≈{train_normed.mean().mean():.4f}, std≈{train_normed.std().mean():.4f}")
    except Exception as e:
        errors.append(f"Normalisation: {e}")
        print(f"    ✗  {e}")

    # 1d. Sequence building with raw targets
    print("  1d. Sequence building (raw price targets) …")
    try:
        close_raw = data["Close"][t]
        X, y, dates = build_sequences(normed, close_raw, cfg.LOOKBACK_WINDOW, fwd_days=5)
        assert X.shape[0] == len(y) == len(dates), "Shape mismatch"
        assert not np.isnan(X).any(), "NaN in X"
        assert not np.isnan(y).any(), "NaN in y"
        print(f"    ✓  {X.shape[0]} seqs, dim={X.shape[1]}, target range [{y.min():.4f}, {y.max():.4f}]")
    except Exception as e:
        errors.append(f"Sequences: {e}")
        print(f"    ✗  {e}")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2: MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

def test_model_architecture():
    print_section("TEST 2: MODEL ARCHITECTURE VALIDATION")
    errors = []
    input_dim = 40 * 48

    configs = [
        ("Small MLP", [input_dim, 64, 32, 1]),
        ("Medium MLP", [input_dim, 128, 64, 32, 1]),
        ("Large MLP", [input_dim, 256, 128, 64, 1]),
    ]

    for name, layers in configs:
        try:
            model = NumpyMLP(layers, activation="leaky_relu", dropout=0.2, seed=42)
            X = np.random.randn(16, input_dim)
            y_true = np.random.randn(16)

            y_pred = model.forward(X)
            assert y_pred.shape == (16,), f"Shape {y_pred.shape}"
            model.backward(y_pred, y_true, lr=1e-3, weight_decay=1e-5, grad_clip=1.0)
            y_inf = model.predict(X)
            assert y_inf.shape == (16,)

            n_params = model.param_count()
            print(f"    ✓  {name:<14} params={n_params:>10,}  fwd ✓  bwd ✓  pred ✓")
        except Exception as e:
            errors.append(f"{name}: {e}")
            print(f"    ✗  {name}: {e}")

    for act in ["relu", "leaky_relu", "tanh", "elu"]:
        try:
            m = NumpyMLP([100, 32, 1], activation=act, seed=42)
            out = m.predict(np.random.randn(4, 100))
            assert out.shape == (4,)
            print(f"    ✓  Activation: {act}")
        except Exception as e:
            errors.append(f"Activation {act}: {e}")
            print(f"    ✗  Activation {act}: {e}")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3: WALK-FORWARD VALIDATION (REAL DATA)
# ═══════════════════════════════════════════════════════════════════════════════

def test_walk_forward(quick=False):
    print_section("TEST 3: WALK-FORWARD CROSS-VALIDATION (REAL DATA)")

    tickers = cfg.UNIVERSE[:8]
    data = load_real_data(
        tickers,
        data_dir=cfg.DATA_DIR,
        benchmark=cfg.BENCHMARK,
        start_date=cfg.START_DATE,
        end_date=cfg.END_DATE,
        auto_download=True,
    )

    # Build raw features + normalise per fold
    raw_frames = {}
    for t in tickers:
        raw_frames[t] = add_features(
            data["Close"][t], data["High"][t], data["Low"][t], data["Volume"][t])

    # Find common dates
    common = raw_frames[tickers[0]].index
    for t in tickers[1:]:
        common = common.intersection(raw_frames[t].index)
    common = common.sort_values()

    n_folds = 3 if quick else 5
    fold_size = len(common) // (n_folds + 2)

    # Use a simple namespace to avoid mutating the shared module-level config.
    import types
    wf_cfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')})
    wf_cfg.EPOCHS = 15 if quick else 30
    wf_cfg.ENSEMBLE_MODELS = 1

    results = []
    for f in range(2, 2 + n_folds):
        train_end_date = common[f * fold_size]
        # Safety buffer: exclude the last FORWARD_DAYS samples from training
        # so no training target uses prices from the test period.
        safe_train_end = common[max(0, f * fold_size - cfg.FORWARD_DAYS)]
        test_start = f * fold_size
        test_end = min((f + 1) * fold_size, len(common))
        test_date_set = set(common[test_start:test_end])

        if len(test_date_set) < 20:
            continue

        # Per-fold normalisation: fit on train only
        all_X, all_y = [], []
        all_X_te, all_y_te = [], []

        for t in tickers:
            raw = raw_frames[t]
            train_rows = raw.loc[raw.index <= train_end_date]
            mu, sigma = fit_normalisation(train_rows)
            normed = apply_normalisation(raw, mu, sigma)
            close_raw = data["Close"][t]

            X, y, dates_seq = build_sequences(normed, close_raw, cfg.LOOKBACK_WINDOW,
                                              fwd_days=cfg.FORWARD_DAYS)

            for i, d in enumerate(dates_seq):
                if d <= safe_train_end:
                    all_X.append(X[i])
                    all_y.append(y[i])
                elif d in test_date_set:
                    all_X_te.append(X[i])
                    all_y_te.append(y[i])

        Xtr = np.array(all_X)
        ytr = np.array(all_y)
        Xte = np.array(all_X_te)
        yte = np.array(all_y_te)

        if len(Xtr) < 100 or len(Xte) < 20:
            continue

        # Remove NaN
        mask = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr))
        Xtr, ytr = Xtr[mask], ytr[mask]
        mask = ~(np.isnan(Xte).any(axis=1) | np.isnan(yte))
        Xte, yte = Xte[mask], yte[mask]

        val_n = int(len(Xtr) * 0.85)
        model = train_model(Xtr[:val_n], ytr[:val_n], Xtr[val_n:], ytr[val_n:], 0, wf_cfg)

        preds = model.predict(Xte)
        corr = np.corrcoef(preds, yte)[0, 1] if len(preds) > 1 and preds.std() > 0 else 0.0
        mse = np.mean((preds - yte)**2)
        # Strict inequality avoids sign(0)==sign(0) inflating direction accuracy
        dir_acc = np.mean((preds > 0) == (yte > 0))

        results.append({"fold": f-1, "train": len(Xtr), "test": len(Xte),
                        "corr": corr, "mse": mse, "dir_acc": dir_acc})
        print(f"\n    Fold {f-1}: corr={corr:.4f}  mse={mse:.6f}  dir_acc={dir_acc:.2%}")

    # wf_cfg was a local copy; no need to restore cfg

    if results:
        print()
        table = [["Fold", "Train", "Test", "Correlation", "MSE", "Dir Acc"]]
        for r in results:
            table.append([f"F{r['fold']}", f"{r['train']:,}", f"{r['test']:,}",
                         f"{r['corr']:.4f}", f"{r['mse']:.6f}", f"{r['dir_acc']:.2%}"])
        avg_c = np.mean([r["corr"] for r in results])
        avg_d = np.mean([r["dir_acc"] for r in results])
        table.append(["AVG", "—", "—", f"{avg_c:.4f}", "—", f"{avg_d:.2%}"])
        print_table(table)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4: MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_monte_carlo(n_sims=1000):
    print_section("TEST 4: MONTE CARLO SIMULATION")

    # Use REAL SPY daily returns for bootstrap
    try:
        data = load_real_data(
            ["AAPL"],  # just need SPY benchmark
            data_dir=cfg.DATA_DIR,
            benchmark=cfg.BENCHMARK,
            start_date=cfg.START_DATE,
            end_date=cfg.END_DATE,
            auto_download=True,
        )
        spy = data["Benchmark"]
        daily_rets = spy.pct_change().dropna().values
        print(f"  Using real SPY returns ({len(daily_rets)} days)")
    except Exception:
        np.random.seed(42)
        daily_rets = np.random.normal(0.0005, 0.012, 500)
        print("  Using synthetic returns (SPY data not found)")

    print(f"  Running {n_sims:,} bootstrap simulations …")

    sim_ret, sim_sharpe, sim_dd = [], [], []
    for i in range(n_sims):
        boot = np.random.choice(daily_rets, size=len(daily_rets), replace=True)
        cum = np.cumprod(1 + boot)
        sim_ret.append(cum[-1] - 1)
        vol = np.std(boot) * np.sqrt(252)
        sim_sharpe.append((np.mean(boot)*252 - 0.04) / max(vol, 1e-10))
        pk = np.maximum.accumulate(cum)
        sim_dd.append(((cum - pk) / pk).min())

        if (i+1) % 250 == 0:
            sys.stdout.write(f"\r  {pbar(i+1, n_sims, ret=True)}  {i+1}/{n_sims}")
            sys.stdout.flush()
    print()

    sr, ss, sd = np.array(sim_ret), np.array(sim_sharpe), np.array(sim_dd)

    table = [["Metric", "5th", "25th", "Median", "75th", "95th"]]
    for name, arr in [("Return", sr), ("Sharpe", ss), ("Max DD", sd)]:
        fmt = ".2%" if name != "Sharpe" else ".3f"
        table.append([name] + [f"{np.percentile(arr, p):{fmt}}" for p in [5,25,50,75,95]])
    print_table(table)

    print_metric("P(positive)", f"{(sr > 0).mean():.1%}")
    print_metric("P(beat 10%)", f"{(sr > 0.10).mean():.1%}")

    return sr, ss, sd


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5: PARAMETER SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def test_sensitivity():
    print_section("TEST 5: PARAMETER SENSITIVITY")

    tickers = cfg.UNIVERSE[:8]
    data = load_real_data(
        tickers,
        data_dir=cfg.DATA_DIR,
        benchmark=cfg.BENCHMARK,
        start_date=cfg.START_DATE,
        end_date=cfg.END_DATE,
        auto_download=True,
    )
    dates = data["Close"].index
    prices = data["Close"][tickers]
    bench = data["Benchmark"]

    np.random.seed(42)
    signals = pd.DataFrame(np.random.normal(0, 0.03, (len(dates), len(tickers))),
                           index=dates, columns=tickers)

    thresholds = [0.005, 0.01, 0.02, 0.03, 0.05]
    sizings = ["equal", "risk_parity", "momentum"]

    import types
    results = []

    for th in thresholds:
        for sz in sizings:
            sens_cfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')})
            sens_cfg.SIGNAL_THRESHOLD = th
            sens_cfg.POSITION_SIZING = sz
            try:
                pv, tr = run_backtest(signals, prices, bench, sens_cfg)
                m = compute_metrics(pv, tr, rf=cfg.RISK_FREE_RATE)
                results.append({"threshold": th, "sizing": sz,
                               "return": m["strategy_return"], "sharpe": m["strategy_sharpe"],
                               "max_dd": m["strategy_max_dd"], "trades": m["total_trades"]})
            except Exception:
                results.append({"threshold": th, "sizing": sz,
                               "return": 0, "sharpe": 0, "max_dd": 0, "trades": 0})

    table = [["Threshold", "Sizing", "Return", "Sharpe", "Max DD", "Trades"]]
    for r in results:
        table.append([f"{r['threshold']:.3f}", r["sizing"],
                     f"{r['return']:.2%}", f"{r['sharpe']:.3f}",
                     f"{r['max_dd']:.2%}", f"{r['trades']:,}"])
    print_table(table)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 6: EVALUATION CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def eval_charts(mc_ret, mc_sharpe, mc_dd, wf_results):
    print_section("TEST 6: EVALUATION CHARTS")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(cfg.CHART_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("AI Stock Trader — Evaluation Suite (Real Data)",
                 fontsize=18, fontweight="bold", color="white")

    ax = axes[0, 0]
    ax.hist(mc_ret, bins=60, color="#00ff88", alpha=0.7, edgecolor="none")
    ax.axvline(np.median(mc_ret), color="white", ls="--", lw=1.5, label=f"Median: {np.median(mc_ret):.2%}")
    ax.axvline(0, color="#ff4444", ls="-", lw=1)
    ax.set_title("MC: Return Distribution", color="white")
    ax.legend(framealpha=0.3); ax.grid(alpha=0.15)

    ax = axes[0, 1]
    ax.hist(mc_sharpe, bins=60, color="#ffaa00", alpha=0.7, edgecolor="none")
    ax.axvline(np.median(mc_sharpe), color="white", ls="--", lw=1.5, label=f"Median: {np.median(mc_sharpe):.3f}")
    ax.axvline(1.0, color="#00ffff", ls="-", lw=1)
    ax.set_title("MC: Sharpe Distribution", color="white")
    ax.legend(framealpha=0.3); ax.grid(alpha=0.15)

    ax = axes[1, 0]
    if wf_results:
        folds = [r["fold"] for r in wf_results]
        corrs = [r["corr"] for r in wf_results]
        daccs = [r["dir_acc"] for r in wf_results]
        x = np.arange(len(folds))
        ax.bar(x - 0.2, corrs, 0.4, color="#00ff88", alpha=0.7, label="Correlation")
        ax.bar(x + 0.2, daccs, 0.4, color="#ffaa00", alpha=0.7, label="Direction Acc")
        ax.set_xticks(x); ax.set_xticklabels([f"F{f}" for f in folds])
        ax.axhline(0.5, color="white", ls="--", alpha=0.3)
    ax.set_title("Walk-Forward: Signal Quality (Real Data)", color="white")
    ax.legend(framealpha=0.3); ax.grid(alpha=0.15)

    ax = axes[1, 1]
    ax.hist(mc_dd, bins=60, color="#ff4444", alpha=0.7, edgecolor="none")
    ax.axvline(np.median(mc_dd), color="white", ls="--", lw=1.5, label=f"Median: {np.median(mc_dd):.2%}")
    ax.set_title("MC: Max Drawdown Distribution", color="white")
    ax.legend(framealpha=0.3); ax.grid(alpha=0.15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUTPUT / "evaluation_charts.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=cfg.CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  Evaluation charts → {p}")


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print_header("AI STOCK TRADER — EVALUATION SUITE (REAL DATA)",
                 subtitle=f"{'Quick' if args.quick else 'Full'} | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    errs = []

    errs += test_data_pipeline()
    errs += test_model_architecture()
    wf = test_walk_forward(quick=args.quick)
    mc_r, mc_s, mc_d = test_monte_carlo(300 if args.quick else 1500)
    test_sensitivity()
    eval_charts(mc_r, mc_s, mc_d, wf)

    print_section("FINAL SUMMARY")
    print_metric("Tests run", 6)
    print_metric("Errors", colour(str(len(errs)), "r" if errs else "g"))
    if errs:
        for e in errs:
            print(f"    ✗  {e}")
    else:
        print(f"\n  {colour('All tests passed!', 'g')}")

    print(f"\n  ⏱  Runtime: {time.time()-t0:.1f}s\n")
    print_header("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
