<div align="center">

# AlphaForge

**Pure-NumPy ensemble neural network that beats the S&P 500 — no PyTorch, no TensorFlow, no scikit-learn.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![NumPy](https://img.shields.io/badge/ML%20Engine-Pure%20NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![Zero ML Deps](https://img.shields.io/badge/ML%20Deps-Zero-f97316?style=flat-square)](#)

<br/>

| Metric | AlphaForge | S&P 500 (SPY) | Edge |
|:--|--:|--:|--:|
| **Total Return** | **75.35%** | 61.38% | **+13.97 pp** |
| **CAGR** | **27.27%** | 22.81% | **+4.46 pp** |
| **Sharpe Ratio** | **1.409** | 1.124 | **+0.285** |
| **Sortino Ratio** | **1.917** | 1.447 | **+0.470** |
| **Max Drawdown** | **−16.84%** | −18.76% | **1.92 pp shallower** |
| **Calmar Ratio** | **1.619** | 1.216 | **+0.403** |

*Out-of-sample · 2023–2025 · 10 bps transaction costs · 20 large-cap equities*

</div>

---

## Overview

AlphaForge is a production-quality quantitative trading research framework built entirely on NumPy. It trains an ensemble of multilayer perceptrons on 41 technical indicators across 20 large-cap US equities, backtests against real historical prices from Yahoo Finance, and outputs detailed performance analytics — all with zero machine-learning framework dependencies.

The project is designed around three principles:

- **No look-ahead bias** — normalisation statistics are fit on training data only; targets use raw forward prices
- **Strict temporal separation** — data is split by date, never shuffled across the train/test boundary
- **Real prices only** — every result comes from actual OHLCV data, not synthetic generation

---

## How It Works

```
 Real OHLCV (yfinance)   20 large-cap equities + SPY benchmark
          │
          ▼
 Feature Engineering     41 indicators per stock · per trading day
 ┌────────────────────────────────────────────────────────┐
 │  Returns (1d–20d)   Moving Averages (SMA/EMA 5–50)    │
 │  MACD + Signal      RSI · Bollinger Bands · ATR        │
 │  Stochastic · CCI   OBV · MFI · Vol-of-Vol · Momentum │
 └────────────────────────────────────────────────────────┘
          │
          ▼
 Temporal Split          70% train  /  30% test  (date-ordered, no shuffle)
 Z-score Normalisation   fit on training rows only → applied to both splits
          │
          ▼
 Sliding Windows         40 days → 1,640-dim input vector per sample
 Target                  raw forward 5-day price return (un-normalised close)
          │
          ▼
 Ensemble Training       3 × MLP  [1640 → 256 → 128 → 64 → 1]
 ┌────────────────────────────────────────────────────────┐
 │  Optimiser   Adam  (β₁ 0.9, β₂ 0.999)                 │
 │  Loss        Huber  (robust to return outliers)        │
 │  Reg.        Dropout 25%  ·  L2 weight decay  ·  grad clip │
 │  Sampling    85% bootstrap per model                   │
 └────────────────────────────────────────────────────────┘
          │
          ▼
 Ensemble Predictions    averaged across 3 models
          │
          ▼
 Portfolio Construction  Softmax weights  ·  60% model / 40% equal
 Execution               Weekly rebalance  ·  always fully invested
 Costs                   10 bps per trade leg
          │
          ▼
 Analytics               Equity curve · Drawdown · Rolling Sharpe
                         Monthly heatmap · Alpha · Risk metrics
```

---

## Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib yfinance
```

AlphaForge's ML engine has **zero additional dependencies**. `yfinance` is used once to download historical data.

### 1 — Download historical data

```bash
python3 - <<'EOF'
import yfinance as yf, pathlib

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "IBM",  "INTC",
    "JPM",  "BAC",  "JNJ",  "UNH",  "XOM",  "PG",   "HD",
    "GS",   "LLY",  "KO",   "MRK",  "WMT",  "PEP",  "SPY",
]

out = pathlib.Path("data")
out.mkdir(exist_ok=True)

for t in TICKERS:
    print(f"  Downloading {t}…")
    df = yf.download(t, start="2004-01-01", end="2025-12-31",
                     auto_adjust=True, progress=False)
    df.to_csv(out / f"{t}.csv")

print("Done.")
EOF
```

### 2 — Train and backtest

```bash
python main.py
```

Runtime: ~3–5 minutes on a modern CPU. No GPU required.

### 3 — Full evaluation suite

```bash
python backtest.py          # walk-forward CV + Monte Carlo + sensitivity analysis
python backtest.py --quick  # fast smoke test (~30 seconds)
```

---

## Project Structure

```
ai-trader/
├── main.py          # entry point — train, backtest, report
├── backtest.py      # evaluation suite — walk-forward, Monte Carlo, sensitivity
├── utils.py         # core library — data, features, MLP, portfolio, charts
├── config.py        # all hyperparameters (edit here to tune)
├── data/            # OHLCV CSVs downloaded via yfinance
└── outputs/
    ├── trading_report.txt   # 8-section performance report
    └── trading_charts.png   # 6-panel dark-theme chart
```

---

## Configuration

All hyperparameters live in `config.py`. No code changes required to tune the model.

```python
# ── Universe ──────────────────────────────────────────────────
UNIVERSE = ["AAPL", "MSFT", "GOOGL", ...]   # stocks to trade
START_DATE = "2004-01-01"                    # data start (warmup period)
TEST_SPLIT = 0.30                            # fraction held out for testing

# ── Feature Engineering ───────────────────────────────────────
LOOKBACK_WINDOW = 40     # days of history per input sample
FORWARD_DAYS    = 5      # prediction horizon (target return window)

# ── Neural Network ────────────────────────────────────────────
HIDDEN_LAYERS   = (256, 128, 64)   # neurons per hidden layer
DROPOUT         = 0.25
ACTIVATION      = "leaky_relu"     # relu | leaky_relu | tanh | elu
EPOCHS          = 50
BATCH_SIZE      = 128
LEARNING_RATE   = 5e-4
WEIGHT_DECAY    = 1e-5
LR_DECAY        = 0.995            # per-epoch multiplicative decay
EARLY_STOP_PATIENCE = 12
GRAD_CLIP       = 2.0

# ── Ensemble ──────────────────────────────────────────────────
ENSEMBLE_MODELS = 3      # models to train and average
BOOTSTRAP_RATIO = 0.85   # fraction of training data per model

# ── Portfolio ─────────────────────────────────────────────────
MAX_POSITION_PCT     = 0.15   # maximum weight per ticker (15%)
TRANSACTION_COST_BPS = 10     # basis points per trade leg
RISK_FREE_RATE       = 0.04   # annualised, for Sharpe / Sortino / Treynor
```

---

## Technical Details

### Neural Network

A full MLP implementation in pure NumPy — no autograd, no frameworks.

| Component | Implementation |
|:--|:--|
| **Forward pass** | Matrix multiply + bias + activation |
| **Backward pass** | Manual chain-rule backpropagation |
| **Optimiser** | Adam (β₁ 0.9, β₂ 0.999, ε 1e-8) |
| **Loss** | Huber (δ=1.0) — robust to fat-tailed return distributions |
| **Regularisation** | Dropout · L2 weight decay · gradient norm clipping |
| **Initialisation** | Xavier/Glorot uniform |
| **Activations** | Leaky ReLU (default) · ReLU · tanh · ELU |
| **Training** | Mini-batch SGD · per-epoch LR decay · early stopping |

### Ensemble Strategy

Each of the 3 models is trained on an 85% bootstrap sample of the training set with a different random seed. Final predictions are the mean across all models, reducing variance and improving out-of-sample stability.

### Portfolio Construction

Signals are converted to weights via a blended softmax scheme:

```
weight_i = 0.40 × (1/N) + 0.60 × softmax(5 × signal_i)
```

Weights are clipped to `[0.005, MAX_POSITION_PCT]` and renormalised to sum to 1.0, keeping the portfolio always fully invested.

---

## Bias-Free Design

Three systematic sources of backtest contamination — and how AlphaForge eliminates each:

| Bias | What goes wrong | AlphaForge's fix |
|:--|:--|:--|
| **Look-ahead in normalisation** | Z-scoring with full-dataset μ/σ leaks future statistics into training features | `fit_normalisation()` computes μ/σ from training rows only; same statistics are applied to the test split |
| **Look-ahead in targets** | Using feature-space differences as targets implicitly encodes future normalisation | Targets are raw forward price returns computed from un-normalised close prices |
| **Temporal leakage** | Shuffling before the train/test split exposes future dates during training | Strict date-ordered split with an additional safety buffer of `FORWARD_DAYS` around the boundary |

---

## Outputs

### Terminal

Live progress during training (per-epoch loss, validation tracking, early-stop indicator) followed by a colour-coded comparison table and final verdict.

### `outputs/trading_report.txt`

Eight-section plaintext report:

1. Executive summary (verdict, alpha, Sharpe)
2. Detailed metrics (CAGR, vol, Sortino, Calmar)
3. Trade statistics (count, win rate, profit factor)
4. Recent trades (last 40, with date / ticker / side / shares / price)
5. Monthly returns (per-month P&L)
6. Drawdown analysis (worst drawdown, recovery time)
7. Risk-adjusted analysis (IR, tracking error, beta, Treynor)
8. Methodology notes

### `outputs/trading_charts.png`

Six-panel dark-theme chart (22×16 in, 150 dpi):

| Panel | Description |
|:--|:--|
| **Equity curve** | Strategy vs SPY, shaded alpha regions |
| **Drawdown** | Underwater plot — strategy and benchmark |
| **Rolling Sharpe** | 63-day annualised Sharpe with positive/negative shading |
| **Monthly heatmap** | Calendar grid coloured by return magnitude |
| **Return distribution** | Daily return histogram vs SPY |
| **Cumulative alpha** | Running outperformance vs benchmark |

---

## Disclaimer

AlphaForge is a **research and educational project**. Backtest results reflect historical simulation under idealised assumptions and do not guarantee future performance. Real-world trading involves risks not captured in backtests — including liquidity constraints, bid-ask spread, market impact, regime changes, and execution latency. This project is not financial advice. Use at your own risk.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute with attribution.
