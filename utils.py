"""
AI Stock Trader — Utilities
==============================
Real stock data loading, bias-free feature engineering,
custom neural network (pure NumPy), portfolio metrics, report generation,
charting, and terminal formatting.

Zero external ML dependencies — only numpy, pandas, matplotlib.

BIAS FIXES vs v1:
  1. add_features() returns RAW (un-normalised) features
  2. fit_normalisation() / apply_normalisation() for train-only Z-scoring
  3. build_sequences() uses actual forward price returns as target
  4. No look-ahead leakage in normalisation or targets
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
#  TERMINAL PRETTY PRINTING
# ═══════════════════════════════════════════════════════════════════════════════

_C = {
    "r": "\033[91m", "g": "\033[92m", "y": "\033[93m", "b": "\033[94m",
    "m": "\033[95m", "c": "\033[96m", "w": "\033[97m",
    "B": "\033[1m", "D": "\033[2m", "0": "\033[0m",
}


def colour(text, c):
    MAP = {"red": "r", "green": "g", "yellow": "y", "blue": "b",
           "magenta": "m", "cyan": "c", "white": "w", "bold": "B", "dim": "D"}
    return f"{_C.get(MAP.get(c, c), '')}{text}{_C['0']}"


def print_header(title, subtitle=""):
    w = 72
    print()
    print(colour("  ╔" + "═" * (w - 2) + "╗", "c"))
    print(colour(f"  ║{title:^{w-2}}║", "c"))
    if subtitle:
        print(colour(f"  ║{subtitle:^{w-2}}║", "c"))
    print(colour("  ╚" + "═" * (w - 2) + "╝", "c"))
    print()


def print_section(title):
    print(f"\n  {colour('─── ' + title + ' ', 'y')}{'─' * max(0, 56 - len(title))}\n")


def print_metric(label, value, w=30):
    print(f"  {colour('•', 'c')} {label:<{w}} {colour(str(value), 'w')}")


def print_table(rows):
    if not rows:
        return
    cols = len(rows[0])
    widths = [max(len(str(rows[r][c])) for r in range(len(rows))) + 2 for c in range(cols)]

    def _row(cells):
        return "  │" + "│".join(f" {str(cells[c]):<{widths[c]}}" for c in range(cols)) + "│"

    top = "  ┌" + "┬".join("─" * (w + 1) for w in widths) + "┐"
    mid = "  ├" + "┼".join("─" * (w + 1) for w in widths) + "┤"
    bot = "  └" + "┴".join("─" * (w + 1) for w in widths) + "┘"

    print(colour(top, "D"))
    print(colour(_row(rows[0]), "B"))
    print(colour(mid, "D"))
    for row in rows[1:]:
        print(_row(row))
    print(colour(bot, "D"))


def pbar(cur, tot, w=30, ret=False):
    pct = cur / max(tot, 1)
    filled = int(w * pct)
    bar = colour("█" * filled, "g") + colour("░" * (w - filled), "D")
    s = f"[{bar}] {pct:>6.1%}"
    if ret:
        return s
    print(s)


# ═══════════════════════════════════════════════════════════════════════════════
#  REAL STOCK DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_real_data(tickers, data_dir="data"):
    """
    Load real OHLCV stock data from CSV files downloaded via yfinance.
    Also loads SPY as the benchmark.

    yfinance auto_adjust=True CSVs have a multi-row header:
      Row 0: Price,Close,High,Low,Open,Volume
      Row 1: Ticker,<sym>,<sym>,<sym>,<sym>,<sym>
      Row 2: Date,,,,,
      Row 3+: data

    Returns the same dict format as the old generate_market_data().
    """
    data_path = Path(data_dir)
    close_frames, high_frames, low_frames, open_frames, vol_frames = {}, {}, {}, {}, {}

    for ticker in tickers:
        csv_path = data_path / f"{ticker}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing data file: {csv_path}")

        df = pd.read_csv(csv_path, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)
        df.index.name = "Date"
        df = df.sort_index()

        # Columns are: Close, High, Low, Open, Volume
        close_frames[ticker] = df["Close"].astype(float)
        high_frames[ticker]  = df["High"].astype(float)
        low_frames[ticker]   = df["Low"].astype(float)
        open_frames[ticker]  = df["Open"].astype(float)
        vol_frames[ticker]   = df["Volume"].astype(float)

    # Align all tickers to common dates
    close_df = pd.DataFrame(close_frames)
    high_df  = pd.DataFrame(high_frames)
    low_df   = pd.DataFrame(low_frames)
    open_df  = pd.DataFrame(open_frames)
    vol_df   = pd.DataFrame(vol_frames)

    # Drop any dates with missing data across tickers
    mask = close_df.notna().all(axis=1)
    close_df = close_df[mask]
    high_df  = high_df.reindex(close_df.index)
    low_df   = low_df.reindex(close_df.index)
    open_df  = open_df.reindex(close_df.index)
    vol_df   = vol_df.reindex(close_df.index)

    # Fill any remaining NaN (holidays etc)
    close_df = close_df.ffill().bfill()
    high_df  = high_df.ffill().bfill()
    low_df   = low_df.ffill().bfill()
    open_df  = open_df.ffill().bfill()
    vol_df   = vol_df.ffill().bfill()

    # Load SPY benchmark
    spy_path = data_path / "SPY.csv"
    if spy_path.exists():
        spy_df = pd.read_csv(spy_path, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)
        spy_df.index.name = "Date"
        spy_df = spy_df.sort_index()
        benchmark = spy_df["Close"].astype(float).reindex(close_df.index).ffill().bfill()
        benchmark.name = "SPY"
    else:
        raise FileNotFoundError(f"Missing benchmark file: {spy_path}")

    return {
        "Open": open_df, "High": high_df, "Low": low_df,
        "Close": close_df, "Volume": vol_df, "Benchmark": benchmark,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING (BIAS-FREE)
# ═══════════════════════════════════════════════════════════════════════════════

def add_features(close, high, low, volume):
    """
    Build a comprehensive feature matrix from OHLCV for one ticker.
    Returns RAW (un-normalised) features — normalisation is done separately
    with train-only statistics to prevent look-ahead bias.
    """
    d = pd.DataFrame(index=close.index)
    c, h, l, v = close, high, low, volume

    # Returns at multiple horizons
    for w in [1, 2, 3, 5, 10, 20]:
        d[f"ret_{w}d"] = c.pct_change(w)
    d["log_ret"] = np.log(c / c.shift(1))

    # Moving averages + slopes
    for w in [5, 10, 20, 50]:
        sma = c.rolling(w).mean()
        d[f"sma{w}_ratio"] = c / sma
        d[f"sma{w}_slope"] = sma.pct_change(5)

    # EMA
    for span in [8, 13, 21]:
        ema = c.ewm(span=span).mean()
        d[f"ema{span}_ratio"] = c / ema

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    sig  = macd.ewm(span=9).mean()
    d["macd"] = macd / c
    d["macd_signal"] = sig / c
    d["macd_hist"] = (macd - sig) / c

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    d["rsi"] = 100 - 100 / (1 + rs)
    d["rsi_norm"] = d["rsi"] / 100 - 0.5

    # Bollinger Bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    d["bb_width"] = (2 * bb_std) / bb_mid.replace(0, 1e-10)
    d["bb_pct"] = (c - (bb_mid - 2*bb_std)) / (4*bb_std).replace(0, 1e-10)

    # ATR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr_pct"] = tr.rolling(14).mean() / c

    # Stochastic
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    d["stoch_k"] = (c - low14) / (high14 - low14).replace(0, 1e-10)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # Williams %R
    d["willr"] = -(high14 - c) / (high14 - low14).replace(0, 1e-10)

    # CCI
    tp = (h + l + c) / 3
    d["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std()).replace(0, 1e-10)
    d["cci_norm"] = d["cci"] / 200

    # OBV (normalised)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    d["obv_norm"] = (obv - obv.rolling(50).mean()) / obv.rolling(50).std().replace(0, 1)

    # MFI
    mf = tp * v
    pos_mf = mf.where(tp > tp.shift(), 0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(), 0).rolling(14).sum()
    d["mfi"] = (100 - 100 / (1 + pos_mf / neg_mf.replace(0, 1e-10))) / 100 - 0.5

    # Volume features
    vsma = v.rolling(20).mean()
    d["vol_ratio"] = v / vsma.replace(0, 1e-10)
    d["vol_trend"] = vsma.pct_change(10)

    # Volatility features
    d["hvol_10"] = d["log_ret"].rolling(10).std() * np.sqrt(252)
    d["hvol_20"] = d["log_ret"].rolling(20).std() * np.sqrt(252)
    d["vol_of_vol"] = d["hvol_10"].rolling(20).std()

    # Mean reversion signal
    d["mean_rev_20"] = -(c / c.rolling(20).mean() - 1)
    d["mean_rev_50"] = -(c / c.rolling(50).mean() - 1)

    # Momentum scores
    d["mom_composite"] = (d["ret_5d"] + d["ret_10d"] + d["ret_20d"]) / 3

    # ── New signal-strengthening features ─────────────────────────────────────

    # Momentum rank (percentile of 20d return in rolling 60-day window)
    d["momentum_rank_20d"] = d["ret_20d"].rolling(60).rank(pct=True)

    # Price channel position — where price sits in its 50-day range [0,1]
    roll_min_50 = c.rolling(50).min()
    roll_max_50 = c.rolling(50).max()
    d["channel_50"] = (c - roll_min_50) / (roll_max_50 - roll_min_50).replace(0, 1e-10)

    # Trend strength (ADX-like) — divergence of fast vs slow MA relative to price
    d["trend_strength"] = abs(c.rolling(20).mean() - c.rolling(50).mean()) / c

    # Gap proxy — high-low range as intraday volatility proxy, normalised
    gap_proxy = (h - l) / c
    d["gap_proxy"] = gap_proxy
    d["gap_proxy_norm"] = gap_proxy / gap_proxy.rolling(20).mean().replace(0, 1e-10)

    # Cumulative return acceleration — is momentum speeding up or slowing down
    d["ret_accel"] = d["ret_5d"] - d["ret_10d"].shift(5)

    # Volatility regime — short vol / long vol ratio (>1 = rising vol regime)
    d["vol_regime"] = d["hvol_10"] / d["hvol_20"].replace(0, 1e-10)

    # Price-volume divergence — price momentum vs volume momentum divergence
    d["pv_diverge"] = d["ret_5d"] - d["vol_ratio"].pct_change(5)

    # ── Trend-following features (strongest alpha in equities) ─────────────

    # 52-week high proximity — stocks near highs tend to continue
    high_252 = c.rolling(252, min_periods=60).max()
    d["near_52w_high"] = c / high_252.replace(0, 1e-10)

    # Risk-adjusted momentum — Sharpe of trailing returns
    ret_20_std = d["log_ret"].rolling(20).std().replace(0, 1e-10)
    d["risk_adj_mom_20"] = d["ret_20d"] / ret_20_std

    # EMA crossover signals — fast vs slow
    ema_5 = c.ewm(span=5).mean()
    ema_20 = c.ewm(span=20).mean()
    d["ema_cross_5_20"] = (ema_5 - ema_20) / c

    # Normalised ATR trend — is volatility expanding or contracting
    d["atr_trend"] = d["atr_pct"].pct_change(10)

    # Clean infinities and NaNs
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.dropna(inplace=True)

    # NO Z-SCORE HERE — returned raw to prevent look-ahead bias
    return d


def fit_normalisation(features_df):
    """Compute normalisation statistics (mean, std) from training data ONLY."""
    mu = features_df.mean()
    sigma = features_df.std().replace(0, 1)
    return mu, sigma


def apply_normalisation(features_df, mu, sigma):
    """Apply pre-computed normalisation statistics to any data split."""
    normed = (features_df - mu) / sigma
    normed.replace([np.inf, -np.inf], 0, inplace=True)
    normed.fillna(0, inplace=True)
    return normed


def build_sequences(features_df, close_series, window, fwd_days=5):
    """
    Create (X, y) sliding-window sequences.

    BIAS FIX: y = actual forward price return over fwd_days, computed from
    raw close prices — NOT from normalised feature values.

    Args:
        features_df: normalised feature DataFrame (already z-scored with train stats)
        close_series: raw close price Series (un-normalised) for target computation
        window: lookback window size
        fwd_days: forward return horizon for target
    """
    vals = features_df.values
    dates = features_df.index.tolist()
    X, y, d_out = [], [], []

    # Align close prices to feature dates
    close_aligned = close_series.reindex(features_df.index)

    for i in range(window, len(vals) - fwd_days):
        X.append(vals[i - window: i].flatten())

        # Target: actual forward return from raw prices (no look-ahead)
        p_now = close_aligned.iloc[i]
        p_fwd = close_aligned.iloc[i + fwd_days]
        fwd_ret = (p_fwd - p_now) / p_now if p_now > 0 else 0.0
        y.append(fwd_ret)
        d_out.append(dates[i])

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), d_out


# ═══════════════════════════════════════════════════════════════════════════════
#  NEURAL NETWORK (Pure NumPy)
# ═══════════════════════════════════════════════════════════════════════════════

class NumpyMLP:
    """
    Multi-layer perceptron trained with Adam, implemented entirely in NumPy.
    Supports: configurable layers, dropout, gradient clipping, L2 regularisation,
    multiple activation functions, and Huber loss.
    """

    def __init__(self, layer_sizes, activation="leaky_relu", dropout=0.25, seed=42):
        self.rng = np.random.RandomState(seed)
        self.layers = []
        self.activation_name = activation
        self.dropout = dropout
        self.training = True

        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            W = self.rng.randn(fan_in, fan_out) * scale
            b = np.zeros((1, fan_out))
            self.layers.append({"W": W, "b": b})

        self.adam_m = [{"W": np.zeros_like(l["W"]), "b": np.zeros_like(l["b"])} for l in self.layers]
        self.adam_v = [{"W": np.zeros_like(l["W"]), "b": np.zeros_like(l["b"])} for l in self.layers]
        self.adam_t = 0

    def _activate(self, z):
        if self.activation_name == "relu":
            return np.maximum(0, z)
        elif self.activation_name == "leaky_relu":
            return np.where(z > 0, z, 0.01 * z)
        elif self.activation_name == "tanh":
            return np.tanh(z)
        elif self.activation_name == "elu":
            return np.where(z > 0, z, 0.5 * (np.exp(np.clip(z, -10, 0)) - 1))
        return z

    def _activate_grad(self, z):
        if self.activation_name == "relu":
            return (z > 0).astype(float)
        elif self.activation_name == "leaky_relu":
            return np.where(z > 0, 1.0, 0.01)
        elif self.activation_name == "tanh":
            t = np.tanh(z)
            return 1 - t ** 2
        elif self.activation_name == "elu":
            return np.where(z > 0, 1.0, 0.5 * np.exp(np.clip(z, -10, 0)))
        return np.ones_like(z)

    def forward(self, X):
        self.cache = {"A": [X], "Z": [], "masks": []}
        A = X
        for i, layer in enumerate(self.layers):
            Z = A @ layer["W"] + layer["b"]
            self.cache["Z"].append(Z)
            if i < len(self.layers) - 1:
                A = self._activate(Z)
                if self.training and self.dropout > 0:
                    mask = (self.rng.rand(*A.shape) > self.dropout).astype(float)
                    A = A * mask / (1 - self.dropout)
                    self.cache["masks"].append(mask)
                else:
                    self.cache["masks"].append(np.ones_like(A))
            else:
                A = Z
            self.cache["A"].append(A)
        return A.flatten() if A.shape[1] == 1 else A

    def backward(self, y_pred, y_true, lr, weight_decay=0, grad_clip=0):
        m = len(y_true)
        diff = y_pred - y_true
        huber_grad = np.where(np.abs(diff) <= 1.0, diff, np.sign(diff))
        dA = huber_grad.reshape(-1, 1) / m

        self.adam_t += 1
        grads = []

        for i in reversed(range(len(self.layers))):
            A_prev = self.cache["A"][i]
            dW = A_prev.T @ dA + weight_decay * self.layers[i]["W"]
            db = dA.sum(axis=0, keepdims=True)
            grads.insert(0, {"W": dW, "b": db})

            if i > 0:
                dA = dA @ self.layers[i]["W"].T
                dA = dA * self._activate_grad(self.cache["Z"][i - 1])
                if self.dropout > 0:
                    dA = dA * self.cache["masks"][i - 1] / (1 - self.dropout)

        if grad_clip > 0:
            total_norm = np.sqrt(sum(np.sum(g["W"]**2) + np.sum(g["b"]**2) for g in grads))
            if total_norm > grad_clip:
                sc = grad_clip / (total_norm + 1e-10)
                for g in grads:
                    g["W"] *= sc
                    g["b"] *= sc

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i, g in enumerate(grads):
            for key in ["W", "b"]:
                self.adam_m[i][key] = beta1 * self.adam_m[i][key] + (1 - beta1) * g[key]
                self.adam_v[i][key] = beta2 * self.adam_v[i][key] + (1 - beta2) * g[key] ** 2
                m_hat = self.adam_m[i][key] / (1 - beta1 ** self.adam_t)
                v_hat = self.adam_v[i][key] / (1 - beta2 ** self.adam_t)
                self.layers[i][key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, X):
        self.training = False
        out = self.forward(X)
        self.training = True
        return out

    def get_params(self):
        return [(l["W"].copy(), l["b"].copy()) for l in self.layers]

    def set_params(self, params):
        for i, (W, b) in enumerate(params):
            self.layers[i]["W"] = W.copy()
            self.layers[i]["b"] = b.copy()

    def param_count(self):
        return sum(l["W"].size + l["b"].size for l in self.layers)


def train_model(X_train, y_train, X_val, y_val, model_idx, cfg):
    """Train a single MLP with early stopping."""
    cfg_mod = cfg  # use the passed config module directly

    n_features = X_train.shape[1]
    layers = [n_features] + cfg_mod.HIDDEN_LAYERS + [1]

    model = NumpyMLP(
        layers,
        activation=cfg_mod.ACTIVATION,
        dropout=cfg_mod.DROPOUT,
        seed=42 + model_idx * 7,
    )

    best_val_loss = float("inf")
    patience_ctr = 0
    best_params = None
    lr = cfg_mod.LEARNING_RATE

    n_params = model.param_count()
    print(f"\n  {colour('Model', 'c')} {model_idx + 1}/{cfg_mod.ENSEMBLE_MODELS}  "
          f"(MLP {layers}, params={n_params:,})")

    n = len(X_train)

    for epoch in range(1, cfg_mod.EPOCHS + 1):
        perm = np.random.permutation(n)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, cfg_mod.BATCH_SIZE):
            end = min(start + cfg_mod.BATCH_SIZE, n)
            xb = X_shuf[start:end]
            yb = y_shuf[start:end]

            pred = model.forward(xb)
            diff = pred - yb
            loss = np.mean(np.where(np.abs(diff) <= 1.0, 0.5 * diff**2, np.abs(diff) - 0.5))
            epoch_loss += loss
            batches += 1

            model.backward(pred, yb, lr, cfg_mod.WEIGHT_DECAY, cfg_mod.GRAD_CLIP)

        val_pred = model.predict(X_val)
        val_diff = val_pred - y_val
        val_loss = np.mean(np.where(np.abs(val_diff) <= 1.0, 0.5 * val_diff**2, np.abs(val_diff) - 0.5))

        bar = pbar(epoch, cfg_mod.EPOCHS, w=28, ret=True)
        sys.stdout.write(
            f"\r    {bar}  ep {epoch:>3}/{cfg_mod.EPOCHS}  "
            f"train={epoch_loss/batches:.6f}  val={val_loss:.6f}  lr={lr:.2e}"
        )
        sys.stdout.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            best_params = model.get_params()
        else:
            patience_ctr += 1
            if patience_ctr >= cfg_mod.EARLY_STOP_PATIENCE:
                print(f"\n    ⏹  Early stop at epoch {epoch}")
                break

        lr *= cfg_mod.LR_DECAY

    print(f"\n    ✓  Best val loss: {best_val_loss:.6f}")
    if best_params is not None:
        model.set_params(best_params)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(predictions, prices, benchmark, cfg_mod):
    """
    Portfolio backtest on real stock prices. Supports long-only and long/short.

    Long/short mode:
      - Stocks with positive signals get long weights
      - Stocks with negative signals get short weights (negative shares)
      - Short P&L: profit when price falls, loss when price rises
      - Gross exposure capped at GROSS_LEVERAGE
      - Net exposure kept within NET_EXPOSURE_RANGE

    Risk management (enforced each day):
      - Per-position stop-loss: exits when unrealized loss exceeds STOP_LOSS_PCT
      - Per-position take-profit: exits when unrealized gain exceeds TAKE_PROFIT_PCT
      - Portfolio drawdown circuit breaker: reduces all positions by 50%
        when portfolio drawdown from peak exceeds MAX_PORTFOLIO_DRAWDOWN

    Rebalances weekly. Includes transaction costs.
    """
    tickers = predictions.columns.tolist()
    dates = predictions.index
    n_tickers = len(tickers)
    cost_mult = cfg_mod.TRANSACTION_COST_BPS / 10_000
    allow_short = getattr(cfg_mod, 'ALLOW_SHORT', False)

    initial_capital = 1_000_000.0
    cash = initial_capital
    holdings = {t: 0.0 for t in tickers}  # shares (negative = short)

    # Initial equal-weight long allocation
    equal_w = 1.0 / n_tickers
    for t in tickers:
        price = prices.loc[dates[0], t]
        alloc = initial_capital * equal_w
        holdings[t] = alloc / price
        cash -= alloc

    # Track entry prices for stop-loss / take-profit
    entry_prices = {t: prices.loc[dates[0], t] for t in tickers}

    portfolio_values = []
    trade_log = []
    rebal_counter = 0
    peak_nav = initial_capital  # track peak NAV for drawdown circuit breaker

    for i, date in enumerate(dates):
        if date not in prices.index:
            continue

        # NAV = cash + long value - short value
        # For shorts: holdings[t] < 0, so holdings[t] * price < 0
        # NAV = cash + sum(holdings[t] * price) — works for both long and short
        nav = cash + sum(holdings[t] * prices.loc[date, t] for t in tickers)

        # ── Risk management: per-position stop-loss and take-profit ──
        stop_loss = getattr(cfg_mod, 'STOP_LOSS_PCT', 0.06)
        take_profit = getattr(cfg_mod, 'TAKE_PROFIT_PCT', 0.18)
        for t in tickers:
            if holdings[t] == 0:
                continue
            current_price = prices.loc[date, t]
            entry_price = entry_prices[t]
            if holdings[t] > 0:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                trade_value = abs(holdings[t] * current_price)
                trade_cost = trade_value * cost_mult
                cash += holdings[t] * current_price - trade_cost
                side = "STOP" if pnl_pct <= -stop_loss else "TAKE_PROFIT"
                trade_log.append((date, t, side, -holdings[t], current_price))
                holdings[t] = 0.0
                entry_prices[t] = current_price  # reset for next entry

        # Recompute NAV after any stop-loss / take-profit exits
        nav = cash + sum(holdings[t] * prices.loc[date, t] for t in tickers)

        # ── Risk management: portfolio-level max drawdown circuit breaker ──
        peak_nav = max(peak_nav, nav)
        max_dd_limit = getattr(cfg_mod, 'MAX_PORTFOLIO_DRAWDOWN', 0.12)
        current_dd = (peak_nav - nav) / peak_nav
        if current_dd > max_dd_limit:
            for t in tickers:
                reduce = holdings[t] * 0.5
                if abs(reduce * prices.loc[date, t]) > nav * 0.001:
                    cash += reduce * prices.loc[date, t]
                    trade_cost = abs(reduce * prices.loc[date, t]) * cost_mult
                    cash -= trade_cost
                    holdings[t] -= reduce
                    trade_log.append((date, t, "DELEVERAGE", -reduce, prices.loc[date, t]))
            peak_nav = nav  # reset peak after deleveraging

        rebal_counter += 1
        rebal_freq = getattr(cfg_mod, 'REBALANCE_DAYS', 5)
        if rebal_counter >= rebal_freq and i < len(dates) - 1:
            rebal_counter = 0
            signals = predictions.loc[date]
            target_w = _signal_to_weights(signals, tickers, cfg_mod)

            for t in tickers:
                price = prices.loc[date, t]
                target_dollar = nav * target_w[t]  # negative for shorts
                target_shares = target_dollar / price
                diff_shares = target_shares - holdings[t]

                if abs(diff_shares * price) > nav * 0.01:
                    trade_cost = abs(diff_shares * price) * cost_mult
                    cash -= trade_cost

                    # Update cash: selling shares gives cash, buying costs cash
                    # For shorts: selling shares we don't own gives cash (short proceeds)
                    cash -= diff_shares * price
                    holdings[t] = target_shares

                    if diff_shares > 0:
                        side = "BUY" if target_shares > 0 else "COVER"
                    else:
                        side = "SELL" if holdings[t] >= 0 else "SHORT"
                    trade_log.append((date, t, side, diff_shares, price))
                    # Update entry price on rebalance (new position or direction change)
                    entry_prices[t] = price

        nav = cash + sum(holdings[t] * prices.loc[date, t] for t in tickers)
        portfolio_values.append({"date": date, "portfolio": nav})

    pv = pd.DataFrame(portfolio_values).set_index("date")
    pv["benchmark"] = benchmark.reindex(pv.index).ffill()
    pv["benchmark"] = pv["benchmark"] / pv["benchmark"].iloc[0] * initial_capital

    trades = pd.DataFrame(trade_log, columns=["date", "ticker", "side", "shares", "price"])
    return pv, trades


def _signal_to_weights(signals, tickers, cfg_mod):
    """
    Convert model signals to portfolio weights.

    Long-only mode (ALLOW_SHORT=False):
        Softmax over all signals → all positive weights summing to 1.0

    Long/short mode (ALLOW_SHORT=True):
        - Positive signals → long weights
        - Negative signals → short weights (negative values)
        - Net exposure kept within NET_EXPOSURE_RANGE
        - Gross exposure capped at GROSS_LEVERAGE
    """
    n = len(tickers)
    scores = np.array([signals.get(t, 0.0) for t in tickers])
    allow_short = getattr(cfg_mod, 'ALLOW_SHORT', False)

    if not allow_short:
        # ── Long-only: softmax weights ────────────────────────────
        base_w = 1.0 / n
        scale = getattr(cfg_mod, 'SOFTMAX_SCALE', 5.0)
        scores_scaled = np.clip(scale * scores, -20, 20)
        exp_scores = np.exp(scores_scaled)
        model_weights = exp_scores / exp_scores.sum()

        blend = getattr(cfg_mod, 'SIGNAL_BLEND', 0.60)
        final_weights = (1 - blend) * base_w + blend * model_weights
        final_weights = np.clip(final_weights, 0.005, cfg_mod.MAX_POSITION_PCT)
        final_weights = final_weights / final_weights.sum()
        return {t: w for t, w in zip(tickers, final_weights)}

    # ── Long/Short mode ───────────────────────────────────────────
    # Strategy: start from the proven long-only softmax allocation,
    # then add a small short overlay on the bottom-ranked stocks.
    # This preserves the long book's alpha while extracting extra
    # returns from shorting the weakest signals.
    short_scale = getattr(cfg_mod, 'SHORT_SCALE', 0.20)
    max_short = getattr(cfg_mod, 'MAX_SHORT_PCT', 0.06)
    gross_cap = getattr(cfg_mod, 'GROSS_LEVERAGE', 1.20)
    net_min, net_max = getattr(cfg_mod, 'NET_EXPOSURE_RANGE', (0.70, 1.10))
    n_short = getattr(cfg_mod, 'N_SHORT', 4)  # only short bottom N stocks

    # ── Step 1: Build long book using proven softmax approach ──
    base_w = 1.0 / n
    scale = getattr(cfg_mod, 'SOFTMAX_SCALE', 5.0)
    scores_scaled = np.clip(scale * scores, -20, 20)
    exp_scores = np.exp(scores_scaled)
    model_weights = exp_scores / exp_scores.sum()

    blend = getattr(cfg_mod, 'SIGNAL_BLEND', 0.60)
    long_weights = (1 - blend) * base_w + blend * model_weights
    long_weights = np.clip(long_weights, 0.005, cfg_mod.MAX_POSITION_PCT)
    long_weights = long_weights / long_weights.sum()

    # ── Step 2: Build small short overlay on worst-ranked stocks ──
    short_weights = np.zeros(n)
    rank_order = np.argsort(scores)  # indices sorted worst→best
    bottom_idx = rank_order[:n_short]

    # Use signal deviation below median as short conviction
    median_score = np.median(scores)
    for idx in bottom_idx:
        deviation = median_score - scores[idx]
        if deviation > 0:
            short_weights[idx] = deviation

    # Normalise short weights to sum to short_scale
    sw_sum = short_weights.sum()
    if sw_sum > 1e-10:
        short_weights = short_weights / sw_sum * short_scale
        # Cap individual short positions
        short_weights = np.clip(short_weights, 0, max_short)
        # Re-normalise after clipping
        sw_sum2 = short_weights.sum()
        if sw_sum2 > short_scale:
            short_weights = short_weights / sw_sum2 * short_scale

    # ── Step 3: Combine long and short books ──
    final_weights = long_weights - short_weights

    # Enforce gross exposure cap
    gross = np.abs(final_weights).sum()
    if gross > gross_cap:
        final_weights = final_weights * (gross_cap / gross)

    # Enforce net exposure range
    net = final_weights.sum()
    if net < net_min:
        deficit = net_min - net
        long_idx = final_weights > 0
        if long_idx.any():
            final_weights[long_idx] += deficit * (final_weights[long_idx] / final_weights[long_idx].sum())
    elif net > net_max:
        excess = net - net_max
        short_idx = final_weights < 0
        if short_idx.any():
            final_weights[short_idx] -= excess * (np.abs(final_weights[short_idx]) / np.abs(final_weights[short_idx]).sum())

    return {t: w for t, w in zip(tickers, final_weights)}


# ═══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(portfolio, trades):
    strat = portfolio["portfolio"]
    bench = portfolio["benchmark"]
    days = len(strat)
    years = days / 252
    rf = 0.04

    strat_ret = strat.iloc[-1] / strat.iloc[0] - 1
    bench_ret = bench.iloc[-1] / bench.iloc[0] - 1

    sd = strat.pct_change().dropna()
    bd = bench.pct_change().dropna()

    sv = sd.std() * np.sqrt(252)
    bv = bd.std() * np.sqrt(252)

    sc = (1 + strat_ret) ** (1 / max(years, 0.01)) - 1
    bc = (1 + bench_ret) ** (1 / max(years, 0.01)) - 1

    ss = (sd.mean() * 252 - rf) / max(sv, 1e-10)
    bs = (bd.mean() * 252 - rf) / max(bv, 1e-10)

    sn = sd[sd < 0].std() * np.sqrt(252)
    bn = bd[bd < 0].std() * np.sqrt(252)
    s_sort = (sd.mean() * 252 - rf) / max(sn, 1e-10)
    b_sort = (bd.mean() * 252 - rf) / max(bn, 1e-10)

    def mdd(s):
        pk = s.cummax()
        return ((s - pk) / pk).min()

    sm = mdd(strat)
    bm = mdd(bench)

    s_cal = sc / abs(sm) if sm != 0 else 0
    b_cal = bc / abs(bm) if bm != 0 else 0

    nt = len(trades)
    if nt > 0:
        tv = abs(trades["shares"] * trades["price"])
        ar = tv.mean() / max(strat.iloc[0], 1)
    else:
        ar = 0

    # Win rate and profit factor from daily portfolio returns
    # (trade log lacks per-position entry/exit P&L for true win rate)
    pos_days = sd[sd > 0]
    neg_days = sd[sd < 0]
    wr = (sd > 0).mean()
    gp = pos_days.sum()
    gl = abs(neg_days.sum())
    pf = gp / max(gl, 1e-10)

    rs = (sd.rolling(63).mean() * 252 - rf) / (sd.rolling(63).std() * np.sqrt(252)).replace(0, 1)
    tracking_diff = sd - bd.reindex(sd.index).fillna(0)
    tracking_error = max(tracking_diff.std() * np.sqrt(252), 0.001)

    return {
        "strategy_return": strat_ret, "benchmark_return": bench_ret,
        "strategy_cagr": sc, "benchmark_cagr": bc,
        "strategy_sharpe": ss, "benchmark_sharpe": bs,
        "strategy_sortino": s_sort, "benchmark_sortino": b_sort,
        "strategy_max_dd": sm, "benchmark_max_dd": bm,
        "strategy_vol": sv, "benchmark_vol": bv,
        "strategy_calmar": s_cal, "benchmark_calmar": b_cal,
        "win_rate": wr, "profit_factor": pf,
        "total_trades": nt, "avg_trade_return": ar,
        "tracking_error": tracking_error,
        "rolling_sharpe": rs, "strat_daily": sd, "bench_daily": bd,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(metrics, trades, portfolio, path, tickers=None, cfg_mod=None):
    import itertools

    L = []
    w = 80
    border = "=" * w
    L.append(border)
    L.append("")
    L.append("A L P H A F O R G E   A I".center(w))
    L.append("Real Data Backtest Report".center(w))
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(w))
    L.append("")
    L.append(border)

    beat = metrics["strategy_return"] > metrics["benchmark_return"]
    alpha = metrics["strategy_return"] - metrics["benchmark_return"]

    # ── Performance Grade ──────────────────────────────────────────────────
    sharpe = metrics["strategy_sharpe"]
    alpha_pct = alpha * 100
    if sharpe > 1.5 and alpha_pct > 10:
        grade = "A+"
    elif sharpe > 1.0 and alpha_pct > 5:
        grade = "A"
    elif beat:
        grade = "B+"
    elif abs(alpha_pct) < 5:
        grade = "B"
    else:
        grade = "C"

    L.append("\n" + "-" * w)
    L.append("1. EXECUTIVE SUMMARY".center(w))
    L.append("-" * w)
    L.append(f"  Verdict:              {'>>> BEAT <<<' if beat else 'UNDERPERFORMED'} the S&P 500")
    L.append(f"  Performance Grade:    {grade}")
    L.append(f"  Alpha:                {alpha:+.2%}")
    L.append(f"  Strategy Return:      {metrics['strategy_return']:.2%}")
    L.append(f"  Benchmark Return:     {metrics['benchmark_return']:.2%}")
    L.append(f"  Strategy Sharpe:      {metrics['strategy_sharpe']:.3f}")
    L.append(f"  Benchmark Sharpe:     {metrics['benchmark_sharpe']:.3f}")

    if tickers:
        L.append(f"\n  Universe:             {', '.join(tickers)}")
        L.append(f"  Number of stocks:     {len(tickers)}")

    # ── Detailed Metrics ───────────────────────────────────────────────────
    L.append("\n" + "-" * w)
    L.append("2. DETAILED METRICS".center(w))
    L.append("-" * w)
    tracking_error = metrics.get('tracking_error', max(abs(metrics['strategy_vol'] - metrics['benchmark_vol']), 0.01))
    info_ratio = alpha / max(tracking_error, 0.001)
    detail_rows = [
        ("Strategy CAGR", "strategy_cagr"),
        ("Benchmark CAGR", "benchmark_cagr"),
        ("Strategy Volatility", "strategy_vol"),
        ("Benchmark Volatility", "benchmark_vol"),
        ("Strategy Sortino", "strategy_sortino"),
        ("Benchmark Sortino", "benchmark_sortino"),
        ("Strategy Max DD", "strategy_max_dd"),
        ("Benchmark Max DD", "benchmark_max_dd"),
        ("Strategy Calmar", "strategy_calmar"),
        ("Benchmark Calmar", "benchmark_calmar"),
    ]
    for label, key in detail_rows:
        v = metrics[key]
        L.append(f"  {label:<28} {v:>12.4f}")
    L.append(f"  {'Tracking Error':<28} {tracking_error:>12.4f}")
    L.append(f"  {'Information Ratio':<28} {info_ratio:>12.3f}")

    # ── Trade Statistics ───────────────────────────────────────────────────
    L.append("\n" + "-" * w)
    L.append("3. TRADE STATISTICS".center(w))
    L.append("-" * w)
    L.append(f"  Total Trades:         {metrics['total_trades']:,}")
    L.append(f"  Win Rate:             {metrics['win_rate']:.1%}")
    L.append(f"  Profit Factor:        {metrics['profit_factor']:.2f}")
    L.append(f"  Avg Trade Return:     {metrics['avg_trade_return']:.4%}")

    # Win/loss streaks
    daily_rets = metrics.get('strat_daily', None)
    if daily_rets is not None and len(daily_rets) > 0:
        streaks_binary = (daily_rets > 0).astype(int)
        max_win_streak = max(
            (sum(1 for _ in g) for k, g in itertools.groupby(streaks_binary) if k == 1),
            default=0,
        )
        max_loss_streak = max(
            (sum(1 for _ in g) for k, g in itertools.groupby(streaks_binary) if k == 0),
            default=0,
        )
        avg_win = daily_rets[daily_rets > 0].mean() * 100
        avg_loss = daily_rets[daily_rets < 0].mean() * 100
        L.append(f"  Max Winning Streak:   {max_win_streak} days")
        L.append(f"  Max Losing Streak:    {max_loss_streak} days")
        L.append(f"  Avg Winning Day:      +{avg_win:.3f}%")
        L.append(f"  Avg Losing Day:       {avg_loss:.3f}%")
        L.append(f"  Win/Loss Magnitude:   {abs(avg_win / avg_loss):.2f}x")

    if len(trades) > 0:
        L.append("\n" + "-" * w)
        L.append("4. RECENT TRADES (last 40)".center(w))
        L.append("-" * w)
        L.append(f"  {'Date':<12} {'Ticker':<8} {'Side':<14} {'Shares':>12} {'Price':>12}")
        L.append(f"  {'─'*12} {'─'*8} {'─'*14} {'─'*12} {'─'*12}")
        for _, row in trades.tail(40).iterrows():
            dt = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10]
            L.append(f"  {dt:<12} {row['ticker']:<8} {row['side']:<14} {row['shares']:>12.2f} ${row['price']:>11.2f}")

    # ── Monthly Returns ────────────────────────────────────────────────────
    L.append("\n" + "-" * w)
    L.append("5. MONTHLY RETURNS".center(w))
    L.append("-" * w)
    strat = portfolio["portfolio"]
    monthly = strat.resample("ME").last().pct_change().dropna()
    L.append(f"  {'Month':<10} {'Return':>10}")
    L.append(f"  {'─'*10} {'─'*10}")
    for dt, ret in monthly.items():
        L.append(f"  {dt.strftime('%Y-%m'):<10} {ret:>+10.2%}")

    # ── Drawdown Analysis ──────────────────────────────────────────────────
    L.append("\n" + "-" * w)
    L.append("6. DRAWDOWN ANALYSIS".center(w))
    L.append("-" * w)
    pk = strat.cummax()
    dd = (strat - pk) / pk
    worst = dd.idxmin()
    L.append(f"  Worst Drawdown:       {dd.min():.2%} on {worst.date()}")
    post = dd[worst:]
    rec = post[post >= 0]
    if len(rec) > 0:
        rd = (rec.index[0] - worst).days
        L.append(f"  Recovery:             {rd} days")
    else:
        L.append(f"  Recovery:             Not recovered")

    # ── Risk-Adjusted Analysis ─────────────────────────────────────────────
    L.append("\n" + "-" * w)
    L.append("7. RISK-ADJUSTED ANALYSIS".center(w))
    L.append("-" * w)
    L.append(f"  Information Ratio:    {info_ratio:.3f}")
    L.append(f"  Tracking Error:       {tracking_error:.4f}")
    L.append(f"  Treynor Ratio:        {(metrics['strategy_cagr'] - 0.04):.4f}")
    L.append(f"  Return/MaxDD:         {abs(metrics['strategy_return'] / min(metrics['strategy_max_dd'], -0.001)):.2f}")

    # ── Risk Controls ──────────────────────────────────────────────────────
    L.append("\n" + "-" * w)
    L.append("8. RISK CONTROLS".center(w))
    L.append("-" * w)
    sl = getattr(cfg_mod, 'STOP_LOSS_PCT', 0.06) * 100 if cfg_mod else 6
    tp = getattr(cfg_mod, 'TAKE_PROFIT_PCT', 0.18) * 100 if cfg_mod else 18
    mdd_lim = getattr(cfg_mod, 'MAX_PORTFOLIO_DRAWDOWN', 0.12) * 100 if cfg_mod else 12
    max_pos = getattr(cfg_mod, 'MAX_POSITION_PCT', 0.15) * 100 if cfg_mod else 15
    tx_bps = getattr(cfg_mod, 'TRANSACTION_COST_BPS', 10) if cfg_mod else 10
    rebal = getattr(cfg_mod, 'REBALANCE_DAYS', 5) if cfg_mod else 5
    rebal_label = "Monthly" if rebal >= 20 else "Bi-weekly" if rebal >= 10 else "Weekly"
    allow_short = getattr(cfg_mod, 'ALLOW_SHORT', False) if cfg_mod else False
    mom_w = getattr(cfg_mod, 'MOMENTUM_WEIGHT', 0) if cfg_mod else 0
    mdl_w = getattr(cfg_mod, 'MODEL_WEIGHT', 1) if cfg_mod else 1
    L.append(f"  Stop-Loss:              Active ({sl:.0f}% per position)")
    L.append(f"  Take-Profit:            Active ({tp:.0f}% per position)")
    L.append(f"  Max Drawdown Breaker:   Active ({mdd_lim:.0f}% portfolio level)")
    L.append(f"  Position Limit:         {max_pos:.0f}% max per ticker")
    L.append(f"  Transaction Costs:      {tx_bps} bps per trade")
    L.append(f"  Rebalance Frequency:    {rebal_label} ({rebal} days)")

    # ── Methodology ────────────────────────────────────────────────────────
    L.append("\n" + "-" * w)
    L.append("9. METHODOLOGY".center(w))
    L.append("-" * w)
    L.append("  Data:                 REAL historical prices (yfinance)")
    L.append("  Benchmark:            SPY (S&P 500 ETF)")
    L.append("  Normalisation:        Train-only Z-scoring (no look-ahead)")
    L.append("  Target:               Raw forward 5-day price return")
    L.append("  Model:                Ensemble MLP (pure NumPy, Adam optimiser)")
    L.append(f"  Signal Blend:         {mom_w:.0%} momentum + {mdl_w:.0%} ML model")
    L.append(f"  Rebalancing:          {rebal_label}, fully invested, softmax weights")
    mode = "Long/Short with short overlay" if allow_short else "Long-only"
    L.append(f"  Mode:                 {mode}")
    L.append(f"  Costs:                {tx_bps} bps per trade")

    L.append("\n" + border)
    L.append("")
    L.append("AlphaForge AI | Powered by Neural Ensemble".center(w))
    L.append("END OF REPORT".center(w))
    L.append("")
    L.append(border)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_charts(portfolio, trades, metrics, path, cfg_mod):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.style.use(cfg_mod.CHART_STYLE)
    fig, axes = plt.subplots(3, 2, figsize=(22, 16))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle("AlphaForge AI -- Real Data Backtest (vs S&P 500)",
                 fontsize=20, fontweight="bold", color="white", y=0.98)

    strat = portfolio["portfolio"]
    bench = portfolio["benchmark"]
    sd = metrics["strat_daily"]
    bd = metrics["bench_daily"]

    ax = axes[0, 0]
    ax.plot(strat.index, strat / 1e6, color="#00ff88", lw=2.2, label="AI Strategy")
    ax.plot(bench.index, bench / 1e6, color="#ff6666", lw=1.3, alpha=0.8, label="S&P 500 (SPY)")
    ax.fill_between(strat.index, strat / 1e6, bench / 1e6,
                    where=strat > bench, color="#00ff88", alpha=0.08)
    ax.fill_between(strat.index, strat / 1e6, bench / 1e6,
                    where=strat <= bench, color="#ff4444", alpha=0.08)
    ax.set_title("Equity Curve ($M)", fontsize=13, color="white")
    ax.legend(framealpha=0.3, fontsize=10)
    ax.set_ylabel("Portfolio ($M)")
    ax.grid(alpha=0.15)

    ax = axes[0, 1]
    pk_s = strat.cummax()
    dd_s = (strat - pk_s) / pk_s * 100
    pk_b = bench.cummax()
    dd_b = (bench - pk_b) / pk_b * 100
    ax.fill_between(dd_s.index, dd_s, 0, color="#ff4444", alpha=0.5, label="Strategy")
    ax.fill_between(dd_b.index, dd_b, 0, color="#6666ff", alpha=0.3, label="S&P 500")
    ax.set_title("Underwater Plot (Drawdown %)", fontsize=13, color="white")
    ax.legend(framealpha=0.3)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.15)

    ax = axes[1, 0]
    rs = metrics.get("rolling_sharpe", pd.Series(dtype=float))
    if len(rs) > 0:
        ax.plot(rs.index, rs, color="#ffaa00", lw=1.0)
        ax.fill_between(rs.index, rs, 0, where=rs > 0, color="#00ff88", alpha=0.15)
        ax.fill_between(rs.index, rs, 0, where=rs <= 0, color="#ff4444", alpha=0.15)
        ax.axhline(0, color="white", ls="--", alpha=0.3)
        ax.axhline(1, color="#00ff88", ls=":", alpha=0.4, label="Sharpe=1")
        ax.axhline(2, color="#00ffff", ls=":", alpha=0.4, label="Sharpe=2")
    ax.set_title("Rolling Sharpe (63-day)", fontsize=13, color="white")
    ax.legend(framealpha=0.3)
    ax.grid(alpha=0.15)

    ax = axes[1, 1]
    monthly = strat.resample("ME").last().pct_change().dropna()
    mdf = pd.DataFrame({"r": monthly})
    mdf["year"] = mdf.index.year
    mdf["month"] = mdf.index.month
    piv = mdf.pivot_table(index="year", columns="month", values="r", aggfunc="mean")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    piv.columns = [month_names[c-1] for c in piv.columns]

    if not piv.empty:
        im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=-0.08, vmax=0.08)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, fontsize=8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=8)
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=7,
                            color="black" if abs(v) < 0.04 else "white")
        plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Monthly Returns Heatmap", fontsize=13, color="white")

    ax = axes[2, 0]
    bins = np.linspace(-0.06, 0.06, 100)
    ax.hist(sd, bins=bins, color="#00ff88", alpha=0.6, label="Strategy", density=True)
    ax.hist(bd, bins=bins, color="#ff6666", alpha=0.4, label="S&P 500", density=True)
    ax.axvline(sd.mean(), color="#00ff88", ls="--", lw=2, label=f"μ={sd.mean()*252:.1%} ann.")
    ax.axvline(bd.mean(), color="#ff6666", ls="--", lw=2)
    ax.set_title("Daily Return Distribution", fontsize=13, color="white")
    ax.legend(framealpha=0.3, fontsize=9)
    ax.grid(alpha=0.15)

    ax = axes[2, 1]
    cs = (1 + sd).cumprod()
    cb = (1 + bd).cumprod()
    common = cs.index.intersection(cb.index)
    alpha_cum = (cs.reindex(common) - cb.reindex(common))
    ax.fill_between(alpha_cum.index, alpha_cum * 100, 0,
                    where=alpha_cum >= 0, color="#00ff88", alpha=0.5, label="+Alpha")
    ax.fill_between(alpha_cum.index, alpha_cum * 100, 0,
                    where=alpha_cum < 0, color="#ff4444", alpha=0.5, label="-Alpha")
    ax.axhline(0, color="white", ls="--", alpha=0.3)
    ax.set_title("Cumulative Alpha vs S&P 500 (%)", fontsize=13, color="white")
    ax.set_ylabel("Alpha (%)")
    ax.legend(framealpha=0.3)
    ax.grid(alpha=0.15)

    for row in axes:
        for a in row:
            a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            a.tick_params(axis="x", rotation=30, labelsize=8)

    # Watermark
    fig.text(0.98, 0.01, "AlphaForge AI | Powered by Neural Ensemble",
             ha="right", va="bottom", fontsize=9, color="#555555",
             fontstyle="italic", alpha=0.7)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg_mod.CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
