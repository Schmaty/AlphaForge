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
| **Total Return** | **394.26%** | 226.01% | **+168.26 pp** |
| **CAGR** | **20.82%** | 15.01% | **+5.81 pp** |
| **Sharpe Ratio** | **0.908** | 0.622 | **+0.286** |
| **Sortino Ratio** | **1.107** | 0.749 | **+0.358** |
| **Max Drawdown** | **−29.12%** | −33.72% | **4.60 pp shallower** |
| **Calmar Ratio** | **0.715** | 0.445 | **+0.270** |

*Out-of-sample · 2017–2025 · 10 bps transaction costs · 20 large-cap equities · 5-model ensemble*

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
 Temporal Split          60% train  /  40% test  (date-ordered, no shuffle)
 Z-score Normalisation   fit on training rows only → applied to both splits
          │
          ▼
 Sliding Windows         40 days → 1,640-dim input vector per sample
 Target                  raw forward 3-day price return (un-normalised close)
          │
          ▼
 Ensemble Training       5 × MLP  [1640 → 256 → 128 → 64 → 1]
 ┌────────────────────────────────────────────────────────┐
 │  Optimiser   Adam  (β₁ 0.9, β₂ 0.999)                 │
 │  Loss        Huber  (robust to return outliers)        │
 │  Reg.        Dropout 20%  ·  L2 weight decay  ·  grad clip │
 │  Sampling    88% bootstrap per model                   │
 └────────────────────────────────────────────────────────┘
          │
          ▼
 Ensemble Predictions    averaged across 5 models
          │
          ▼
 Portfolio Construction  Softmax weights  ·  75% model / 25% equal
 Execution               3-day rebalance  ·  always fully invested
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
TEST_SPLIT = 0.40                            # fraction held out for testing

# ── Feature Engineering ───────────────────────────────────────
LOOKBACK_WINDOW = 40     # days of history per input sample
FORWARD_DAYS    = 3      # prediction horizon (target return window)

# ── Neural Network ────────────────────────────────────────────
HIDDEN_LAYERS   = (256, 128, 64)   # neurons per hidden layer
DROPOUT         = 0.20
ACTIVATION      = "leaky_relu"     # relu | leaky_relu | tanh | elu
EPOCHS          = 60
BATCH_SIZE      = 128
LEARNING_RATE   = 6e-4
WEIGHT_DECAY    = 1e-5
LR_DECAY        = 0.993            # per-epoch multiplicative decay
EARLY_STOP_PATIENCE = 15
GRAD_CLIP       = 2.0

# ── Ensemble ──────────────────────────────────────────────────
ENSEMBLE_MODELS = 5      # models to train and average
BOOTSTRAP_RATIO = 0.88   # fraction of training data per model

# ── Portfolio ─────────────────────────────────────────────────
MAX_POSITION_PCT     = 0.20   # maximum weight per ticker (20%)
REBALANCE_DAYS       = 3      # trading days between rebalances
SIGNAL_BLEND         = 0.75   # model signal weight (75% model / 25% equal)
SIGNAL_SCALE         = 8.0    # softmax temperature for weight differentiation
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

Each of the 5 models is trained on an 88% bootstrap sample of the training set with a different random seed. Final predictions are the mean across all models, reducing variance and improving out-of-sample stability.

### Portfolio Construction

Signals are converted to weights via a blended softmax scheme:

```
weight_i = 0.25 × (1/N) + 0.75 × softmax(8 × signal_i)
```

Weights are clipped to `[0.005, MAX_POSITION_PCT]` and renormalised to sum to 1.0, keeping the portfolio always fully invested. The higher blend (75%) and sharper softmax temperature (8.0) allow the model to express stronger conviction in its top picks while maintaining diversification through the equal-weight floor.

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
