# AlphaForge 🔥

> A pure-NumPy AI stock trader that **beat the S&P 500 by +16.44%** on real out-of-sample data — no PyTorch, no TensorFlow, no scikit-learn.

AlphaForge trains an ensemble of neural networks entirely from scratch in NumPy, computes 41 technical indicators per stock, and runs a fully-invested weekly-rebalancing portfolio across 20 large-cap equities. The backtest uses **real historical prices** from Yahoo Finance, a clean temporal train/test split, and no look-ahead bias.

---

## Results (Out-of-Sample, 2023–2025)

| Metric | AlphaForge | S&P 500 (SPY) |
|---|---|---|
| **Total Return** | **73.87%** | 57.43% |
| **CAGR** | **27.06%** | 21.71% |
| **Sharpe Ratio** | **1.374** | 1.068 |
| **Sortino Ratio** | **1.893** | 1.374 |
| **Max Drawdown** | **-16.48%** | -18.76% |
| **Calmar Ratio** | **1.642** | 1.158 |
| Annualised Vol | 15.42% | 15.86% |

*Trained on 2018–2023 real OHLCV data. Tested on unseen 2023–2025 data. 10 bps transaction costs applied.*

---

## How It Works

```
Real OHLCV Data (yfinance)
        │
        ▼
41 Technical Features per stock
(Returns, SMAs, EMAs, MACD, RSI, Bollinger,
 ATR, Stochastic, CCI, OBV, MFI, Vol, Momentum)
        │
        ▼
Temporal Train/Test Split (70/30)
Train-only Z-score Normalisation  ← no look-ahead
        │
        ▼
Sliding Window Sequences (40 days → 1,640-dim input)
Target = raw forward 5-day price return  ← no bias
        │
        ▼
Ensemble of 3 × MLP [1640→256→128→64→1]
Adam Optimiser | Huber Loss | Dropout | Early Stopping
Bootstrap Sampling per model
        │
        ▼
Signal → Softmax Portfolio Weights
60% model tilt + 40% equal weight
Weekly Rebalancing | Always Fully Invested
        │
        ▼
Backtest on Real Prices → vs SPY
```

---

## Project Structure

```
alphaforge/
├── main.py          # Entry point — loads data, trains, backtests, prints results
├── backtest.py      # Evaluation suite — walk-forward CV, Monte Carlo, sensitivity
├── utils.py         # All logic — data loading, features, MLP, backtest engine, charts
├── config.py        # All hyperparameters — edit here to tune the model
├── data/            # Real OHLCV CSVs (downloaded separately, see below)
└── outputs/
    ├── trading_report.txt   # Full text report with metrics and trade log
    └── trading_charts.png   # 6-panel chart: equity curve, drawdown, Sharpe, etc.
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib yfinance
```

AlphaForge's ML engine has **zero dependencies** beyond NumPy. `yfinance` is only needed to download the data once.

### 2. Download real stock data

```bash
python3 -c "
import yfinance as yf, pathlib
tickers = ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','JPM','V','JNJ',
           'UNH','XOM','PG','HD','MA','LLY','ABBV','MRK','AVGO','PEP','SPY']
out = pathlib.Path('data')
out.mkdir(exist_ok=True)
for t in tickers:
    print(f'Downloading {t}...')
    df = yf.download(t, start='2018-01-01', end='2025-12-31', auto_adjust=True, progress=False)
    df.to_csv(out / f'{t}.csv')
print('Done.')
"
```

### 3. Train and backtest

```bash
python main.py
```

### 4. Run the full evaluation suite

```bash
python backtest.py          # Full: walk-forward CV + Monte Carlo + sensitivity
python backtest.py --quick  # Fast smoke test
```

---

## Configuration

Everything is in `config.py`. Key settings:

```python
# Universe
UNIVERSE = ["AAPL", "MSFT", "GOOGL", ...]   # stocks to trade

# Data
START_DATE = "2018-01-01"
END_DATE   = "2025-12-31"
TEST_SPLIT = 0.30                            # 30% held out for testing

# Model
HIDDEN_LAYERS = [256, 128, 64]              # MLP architecture
EPOCHS        = 50
LEARNING_RATE = 5e-4
DROPOUT       = 0.25
ENSEMBLE_MODELS = 3                         # number of models to average

# Portfolio
MAX_POSITION_PCT    = 0.15                  # max weight per stock
TRANSACTION_COST_BPS = 10                   # 10 basis points per trade
LOOKBACK_WINDOW     = 40                    # days of history fed to model
```

---

## Neural Network Details

AlphaForge implements a full MLP from scratch using only NumPy:

- **Architecture**: configurable hidden layers, linear output
- **Optimiser**: Adam (β₁=0.9, β₂=0.999)
- **Loss**: Huber loss (robust to outliers)
- **Regularisation**: dropout, L2 weight decay, gradient clipping
- **Initialisation**: Xavier/Glorot
- **Activations**: leaky ReLU (default), ReLU, tanh, ELU
- **Training**: mini-batch SGD with LR decay and early stopping
- **Ensemble**: 3–5 models with bootstrap sampling, predictions averaged

---

## Bias-Free Design

Three common backtesting mistakes avoided:

| Bias | What it is | How AlphaForge avoids it |
|---|---|---|
| **Look-ahead in normalisation** | Z-scoring with full-dataset stats leaks future info into training | `fit_normalisation()` uses training rows only; same stats applied to test |
| **Look-ahead in target** | Using normalised feature differences as targets encodes future data | Target = raw forward price return from un-normalised close prices |
| **Temporal leakage** | Shuffling before splitting exposes future data during training | Strict date-based split; test period is always strictly after train period |

---

## Outputs

Running `main.py` produces:

**Terminal** — live training progress bars, signal quality stats, and a formatted comparison table vs SPY.

**`outputs/trading_report.txt`** — 8-section report: executive summary, detailed metrics, trade statistics, recent trades, monthly returns, drawdown analysis, risk-adjusted stats, methodology notes.

**`outputs/trading_charts.png`** — 6-panel dark-theme chart:
1. Equity curve vs SPY
2. Underwater (drawdown) plot
3. Rolling 63-day Sharpe ratio
4. Monthly returns heatmap
5. Daily return distribution
6. Cumulative alpha vs SPY

---

## Disclaimer

AlphaForge is a research and educational project. Past backtest performance does not guarantee future results. This is not financial advice. Real trading involves risks not captured in backtests, including liquidity constraints, slippage, and market regime changes. Use at your own risk.

---

## License

MIT
