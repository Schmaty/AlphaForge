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

import re as _re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict

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


def _vis_len(s):
    """Visual length of a string, excluding ANSI escape codes."""
    return len(_re.sub(r'\033\[[0-9;]*m', '', str(s)))


def colour_cmp(ai_val, bm_val, fmt, higher_better=True):
    """Format ai_val coloured green/red based on comparison to bm_val."""
    beats = (ai_val > bm_val) if higher_better else (ai_val < bm_val)
    return colour(format(ai_val, fmt), "g" if beats else "r")


def print_header(title, subtitle=""):
    w = 70
    c, b, d, rst = _C["c"], _C["B"], _C["D"], _C["0"]
    print()
    print(f"  {c}╔{'═' * w}╗{rst}")
    t = title.upper()
    pad_l = (w - len(t)) // 2
    pad_r = w - len(t) - pad_l
    print(f"  {c}║{' ' * pad_l}{b}{t}{rst}{c}{' ' * pad_r}║{rst}")
    if subtitle:
        sp_l = (w - len(subtitle)) // 2
        sp_r = w - len(subtitle) - sp_l
        print(f"  {c}║{' ' * sp_l}{d}{subtitle}{rst}{c}{' ' * sp_r}║{rst}")
    print(f"  {c}╚{'═' * w}╝{rst}")
    print()


def print_section(title):
    fill = max(0, 60 - len(title))
    print(f"\n  {colour('▸', 'y')} {colour(title, 'B')}  {colour('─' * fill, 'D')}\n")


def print_metric(label, value, w=30):
    v = str(value)
    s = v.strip()
    if s.startswith('+'):
        colored_v = colour(v, 'g')
    elif s.startswith('-') and len(s) > 1 and (s[1].isdigit() or s[1] == '.'):
        colored_v = colour(v, 'r')
    else:
        colored_v = colour(v, 'w')
    print(f"  {colour('•', 'c')} {label:<{w}} {colored_v}")


def print_table(rows):
    if not rows:
        return
    cols = len(rows[0])
    # Use _vis_len so ANSI-coded cells don't distort column widths
    widths = [max(_vis_len(rows[r][c]) for r in range(len(rows))) + 2 for c in range(cols)]

    def _row(cells):
        parts = []
        for c in range(cols):
            cell = str(cells[c])
            pad = widths[c] - _vis_len(cell)
            parts.append(f" {cell}{' ' * pad}")
        return "  │" + "│".join(parts) + "│"

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

def _read_yfinance_csv(csv_path):
    """Read a yfinance CSV written by DataFrame.to_csv()."""
    df = pd.read_csv(csv_path, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)
    df.index.name = "Date"
    df = df.sort_index()

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing_cols)}")

    return df


def download_market_data(
    tickers,
    data_dir="data",
    start_date=None,
    end_date=None,
):
    """
    Download OHLCV CSVs from Yahoo Finance for the requested symbols.

    CSVs are written in the same raw yfinance format that load_real_data()
    expects, so the rest of the pipeline can keep using the existing loader.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required to auto-download market data. "
            "Install it with: pip install yfinance"
        ) from exc

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for ticker in tickers:
        print(f"  Downloading {ticker} from Yahoo Finance …")
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to download {ticker}: {exc}") from exc

        if df.empty:
            raise ValueError(
                f"Yahoo Finance returned no rows for {ticker} "
                f"(start={start_date}, end={end_date})."
            )

        df.to_csv(data_path / f"{ticker}.csv")
        downloaded.append(ticker)

    return downloaded


def ensure_market_data(
    tickers,
    data_dir="data",
    benchmark="SPY",
    start_date=None,
    end_date=None,
):
    """Download any missing ticker CSVs required by the pipeline."""
    data_path = Path(data_dir)
    required = list(dict.fromkeys([*tickers, benchmark]))
    missing = []

    for ticker in required:
        csv_path = data_path / f"{ticker}.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            missing.append(ticker)

    if not missing:
        return []

    print(
        f"  {colour('•', 'c')} Missing {len(missing)} data file(s) in "
        f"{data_path}/; downloading required symbols …"
    )
    downloaded = download_market_data(
        missing,
        data_dir=data_path,
        start_date=start_date,
        end_date=end_date,
    )
    print(f"  {colour('✓', 'g')} Downloaded {len(downloaded)} symbol file(s).")
    return downloaded


def load_real_data(
    tickers,
    data_dir="data",
    benchmark="SPY",
    start_date=None,
    end_date=None,
    auto_download=False,
):
    """
    Load real OHLCV stock data from CSV files downloaded via yfinance.

    yfinance auto_adjust=True CSVs have a multi-row header:
      Row 0: Price,Close,High,Low,Open,Volume
      Row 1: Ticker,<sym>,<sym>,<sym>,<sym>,<sym>
      Row 2: Date,,,,,
      Row 3+: data

    If auto_download=True, any missing symbol CSVs are fetched before loading.
    Returns the same dict format as the old generate_market_data().
    """
    data_path = Path(data_dir)
    if auto_download:
        ensure_market_data(
            tickers,
            data_dir=data_path,
            benchmark=benchmark,
            start_date=start_date,
            end_date=end_date,
        )

    close_frames, high_frames, low_frames, open_frames, vol_frames = {}, {}, {}, {}, {}

    for ticker in tickers:
        csv_path = data_path / f"{ticker}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing data file: {csv_path}")

        df = _read_yfinance_csv(csv_path)

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

    # Fill any remaining NaN (holidays etc) — forward-fill only.
    # bfill is intentionally omitted: it would use future prices to fill
    # leading gaps, introducing look-ahead bias.
    close_df = close_df.ffill()
    high_df  = high_df.ffill()
    low_df   = low_df.ffill()
    open_df  = open_df.ffill()
    vol_df   = vol_df.ffill()

    # Load benchmark
    benchmark_ticker = benchmark
    benchmark_path = data_path / f"{benchmark_ticker}.csv"
    if benchmark_path.exists():
        benchmark_df = _read_yfinance_csv(benchmark_path)
        benchmark = benchmark_df["Close"].astype(float).reindex(close_df.index).ffill()
        benchmark.name = benchmark_ticker
    else:
        raise FileNotFoundError(f"Missing benchmark file: {benchmark_path}")

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
    import config as cfg_mod

    n_features = X_train.shape[1]
    layers = [n_features] + list(cfg_mod.HIDDEN_LAYERS) + [1]

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
    arch = " → ".join(str(x) for x in layers)
    print(f"\n  {colour(f'● Model {model_idx + 1}/{cfg_mod.ENSEMBLE_MODELS}', 'c')}  "
          f"{colour(arch, 'B')}  {colour(f'{n_params:,} params', 'D')}")

    n = len(X_train)
    e_w = len(str(cfg_mod.EPOCHS))  # epoch number display width

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

        is_best = val_loss < best_val_loss
        bm = colour("↓", "g") if is_best else " "
        bar = pbar(epoch, cfg_mod.EPOCHS, w=26, ret=True)
        sys.stdout.write(
            f"\r    {colour(f'[{epoch:>{e_w}}/{cfg_mod.EPOCHS}]', 'D')} {bar}  "
            f"{colour('train', 'D')} {epoch_loss/batches:.4e}  "
            f"{colour('val', 'D')} {colour(f'{val_loss:.4e}', 'g' if is_best else 'w')} {bm}  "
            f"{colour('lr', 'D')} {lr:.2e}   "
        )
        sys.stdout.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            best_params = model.get_params()
        else:
            patience_ctr += 1
            if patience_ctr >= cfg_mod.EARLY_STOP_PATIENCE:
                print(f"\n    {colour('⏹', 'y')}  early stop  ep {epoch}  "
                      f"{colour('best val', 'D')} {best_val_loss:.6f}")
                break

        lr *= cfg_mod.LR_DECAY

    print(f"\n    {colour('✓', 'g')}  best val loss {colour(f'{best_val_loss:.6f}', 'w')}")
    model.set_params(best_params)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(predictions, prices, benchmark, cfg_mod):
    """
    Fully-invested portfolio backtest on real stock prices.
    The model determines WEIGHTS across tickers (long-only with tilts).
    Rebalances weekly. Includes transaction costs.
    """
    tickers = predictions.columns.tolist()
    dates = predictions.index
    n_tickers = len(tickers)
    cost_mult = cfg_mod.TRANSACTION_COST_BPS / 10_000

    initial_capital = 1_000_000.0
    holdings = {}
    equal_w = 1.0 / n_tickers
    for t in tickers:
        price = prices.loc[dates[0], t]
        alloc = initial_capital * equal_w
        holdings[t] = alloc / price

    portfolio_values = []
    trade_log = []
    rebal_counter = 0

    for i, date in enumerate(dates):
        if date not in prices.index:
            continue

        port_val = sum(holdings[t] * prices.loc[date, t] for t in tickers)

        rebal_counter += 1
        if rebal_counter >= getattr(cfg_mod, 'REBALANCE_DAYS', 5) and i < len(dates) - 1:
            rebal_counter = 0
            signals = predictions.loc[date]
            target_w = _signal_to_weights(signals, tickers, cfg_mod)

            for t in tickers:
                price = prices.loc[date, t]
                target_shares = (port_val * target_w[t]) / price
                diff_shares = target_shares - holdings[t]

                if abs(diff_shares * price) > port_val * 0.002:
                    trade_cost = abs(diff_shares * price) * cost_mult
                    port_val -= trade_cost
                    holdings[t] = target_shares
                    side = "BUY" if diff_shares > 0 else "SELL"
                    trade_log.append((date, t, side, diff_shares, price))

        port_val = sum(holdings[t] * prices.loc[date, t] for t in tickers)
        portfolio_values.append({"date": date, "portfolio": port_val})

    pv = pd.DataFrame(portfolio_values).set_index("date")
    pv["benchmark"] = benchmark.reindex(pv.index).ffill()
    pv["benchmark"] = pv["benchmark"] / pv["benchmark"].iloc[0] * initial_capital

    trades = pd.DataFrame(trade_log, columns=["date", "ticker", "side", "shares", "price"])
    return pv, trades


def _signal_to_weights(signals, tickers, cfg_mod):
    """
    Convert model signals to portfolio weights.
    Always fully invested (weights sum to 1.0).
    """
    n = len(tickers)
    base_w = 1.0 / n

    scores = np.array([signals.get(t, 0.0) for t in tickers])

    # Softmax-like transformation for weight tilts
    scale = 5.0
    scores_scaled = np.clip(scale * scores, -20, 20)
    exp_scores = np.exp(scores_scaled)
    model_weights = exp_scores / exp_scores.sum()

    # Blend: 40% equal weight + 60% model signal
    blend = 0.60
    final_weights = (1 - blend) * base_w + blend * model_weights

    final_weights = np.clip(final_weights, 0.005, cfg_mod.MAX_POSITION_PCT)
    final_weights = final_weights / final_weights.sum()

    return {t: w for t, w in zip(tickers, final_weights)}


# ═══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(portfolio, trades, rf=0.04):
    strat = portfolio["portfolio"]
    bench = portfolio["benchmark"]
    days = len(strat)
    years = days / 252

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

    # Beta (market sensitivity)
    common_idx = sd.index.intersection(bd.index)
    sd_aligned = sd.reindex(common_idx)
    bd_aligned = bd.reindex(common_idx)
    cov_matrix = np.cov(sd_aligned.values, bd_aligned.values)
    beta = cov_matrix[0, 1] / max(cov_matrix[1, 1], 1e-10)

    # Tracking error and Information Ratio (both annualized)
    active_daily = sd_aligned - bd_aligned
    te = active_daily.std() * np.sqrt(252)
    ir = (sc - bc) / max(te, 1e-10)

    # Treynor Ratio: annualized excess return per unit of market risk
    treynor = (sc - rf) / abs(beta) if abs(beta) > 1e-4 else 0.0

    # Win rate and profit factor based on daily portfolio returns
    # (trade-level P&L is not tracked; daily returns are the reliable measure)
    sd_pos = sd[sd > 0]
    sd_neg = sd[sd < 0]
    wr = len(sd_pos) / max(len(sd), 1)   # fraction of positive-return days
    gp = sd_pos.sum()
    gl = abs(sd_neg.sum())
    pf = gp / max(gl, 1e-10)             # sum(gains) / sum(losses) on daily returns

    nt = len(trades)
    if nt > 0:
        tv = trades["shares"].abs() * trades["price"]
        ar = tv.mean() / max(strat.iloc[0], 1)   # avg trade size as % of initial capital
    else:
        ar = 0.0

    rs = (sd.rolling(63).mean() * 252 - rf) / (sd.rolling(63).std() * np.sqrt(252)).replace(0, 1)

    return {
        "strategy_return": strat_ret, "benchmark_return": bench_ret,
        "strategy_cagr": sc, "benchmark_cagr": bc,
        "strategy_sharpe": ss, "benchmark_sharpe": bs,
        "strategy_sortino": s_sort, "benchmark_sortino": b_sort,
        "strategy_max_dd": sm, "benchmark_max_dd": bm,
        "strategy_vol": sv, "benchmark_vol": bv,
        "strategy_calmar": s_cal, "benchmark_calmar": b_cal,
        "win_rate": wr, "profit_factor": pf,
        "total_trades": nt, "avg_trade_size": ar,
        "beta": beta, "tracking_error": te,
        "information_ratio": ir, "treynor_ratio": treynor,
        "rolling_sharpe": rs, "strat_daily": sd, "bench_daily": bd,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(metrics, trades, portfolio, path, tickers=None):
    L = []
    w = 80
    L.append("=" * w)
    L.append("AI STOCK TRADER — REAL DATA BACKTEST REPORT".center(w))
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(w))
    L.append("=" * w)

    beat = metrics["strategy_return"] > metrics["benchmark_return"]
    alpha = metrics["strategy_return"] - metrics["benchmark_return"]

    L.append("\n1. EXECUTIVE SUMMARY")
    L.append("-" * 60)
    L.append(f"  Verdict:              {'BEAT' if beat else 'UNDERPERFORMED'} the S&P 500")
    L.append(f"  Alpha:                {alpha:+.2%}")
    L.append(f"  Strategy Return:      {metrics['strategy_return']:.2%}")
    L.append(f"  Benchmark Return:     {metrics['benchmark_return']:.2%}")
    L.append(f"  Strategy Sharpe:      {metrics['strategy_sharpe']:.3f}")
    L.append(f"  Benchmark Sharpe:     {metrics['benchmark_sharpe']:.3f}")

    if tickers:
        L.append(f"\n  Universe:             {', '.join(tickers)}")
        L.append(f"  Number of stocks:     {len(tickers)}")

    L.append("\n2. DETAILED METRICS")
    L.append("-" * 60)
    pct_keys = {
        "strategy_cagr", "benchmark_cagr",
        "strategy_vol", "benchmark_vol",
        "strategy_max_dd", "benchmark_max_dd",
    }
    detail_keys = [
        ("Strategy CAGR", "strategy_cagr"), ("Benchmark CAGR", "benchmark_cagr"),
        ("Strategy Volatility", "strategy_vol"), ("Benchmark Volatility", "benchmark_vol"),
        ("Strategy Sortino", "strategy_sortino"), ("Benchmark Sortino", "benchmark_sortino"),
        ("Strategy Max DD", "strategy_max_dd"), ("Benchmark Max DD", "benchmark_max_dd"),
        ("Strategy Calmar", "strategy_calmar"), ("Benchmark Calmar", "benchmark_calmar"),
    ]
    for label, key in detail_keys:
        v = metrics[key]
        if key in pct_keys:
            L.append(f"  {label:<28} {v:>11.2%}")
        else:
            L.append(f"  {label:<28} {v:>12.4f}")

    L.append("\n3. TRADE STATISTICS")
    L.append("-" * 60)
    L.append(f"  Total Trades:         {metrics['total_trades']:,}")
    L.append(f"  Positive Days:        {metrics['win_rate']:.1%}  (daily portfolio return > 0)")
    L.append(f"  Daily Profit Factor:  {metrics['profit_factor']:.2f}  (sum gains / sum losses)")
    L.append(f"  Avg Trade Size:       {metrics['avg_trade_size']:.4%}  (% of initial capital)")

    if len(trades) > 0:
        L.append("\n4. RECENT TRADES (last 40)")
        L.append("-" * 60)
        L.append(f"  {'Date':<12} {'Ticker':<8} {'Side':<6} {'Shares':>10} {'Price ($)':>12}")
        L.append(f"  {'─'*12} {'─'*8} {'─'*6} {'─'*10} {'─'*12}")
        for _, row in trades.tail(40).iterrows():
            dt = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10]
            price_str = f"${row['price']:,.2f}"
            L.append(f"  {dt:<12} {row['ticker']:<8} {row['side']:<6} {abs(row['shares']):>10.2f} {price_str:>12}")

    L.append("\n5. MONTHLY RETURNS")
    L.append("-" * 60)
    strat = portfolio["portfolio"]
    monthly = strat.resample("ME").last().pct_change().dropna()
    L.append(f"  {'Month':<10} {'Return':>10}")
    L.append(f"  {'─'*10} {'─'*10}")
    for dt, ret in monthly.items():
        L.append(f"  {dt.strftime('%Y-%m'):<10} {ret:>+10.2%}")

    L.append("\n6. DRAWDOWN ANALYSIS")
    L.append("-" * 60)
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

    L.append("\n7. RISK-ADJUSTED ANALYSIS")
    L.append("-" * 60)
    L.append(f"  Information Ratio:    {metrics['information_ratio']:.3f}  (ann. active return / tracking error)")
    L.append(f"  Tracking Error:       {metrics['tracking_error']:.2%}  (annualized)")
    L.append(f"  Beta:                 {metrics['beta']:.3f}  (vs SPY)")
    L.append(f"  Treynor Ratio:        {metrics['treynor_ratio']:.3f}  ((CAGR - rf) / beta)")
    L.append(f"  Return/MaxDD:         {abs(metrics['strategy_return'] / min(metrics['strategy_max_dd'], -0.001)):.2f}")

    L.append("\n8. METHODOLOGY")
    L.append("-" * 60)
    L.append("  Data:                 REAL historical prices (yfinance)")
    L.append("  Benchmark:            SPY (S&P 500 ETF)")
    L.append("  Normalisation:        Train-only Z-scoring (no look-ahead)")
    L.append("  Target:               Raw forward 5-day price return")
    L.append("  Model:                Ensemble MLP (pure NumPy, Adam optimiser)")
    L.append("  Rebalancing:          Weekly, fully invested, softmax weights")
    L.append("  Costs:                10 bps per trade")

    L.append("\n" + "=" * w)
    L.append("END OF REPORT".center(w))
    L.append("=" * w)

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
    fig.suptitle("AI Stock Trader — Real Data Backtest (vs S&P 500)",
                 fontsize=20, fontweight="bold", color="white", y=0.98)

    strat = portfolio["portfolio"]
    bench = portfolio["benchmark"]
    sd = metrics["strat_daily"]
    bd = metrics["bench_daily"]

    ax = axes[0, 0]
    ax.plot(strat.index, strat / 1e6, color="#00ff88", lw=1.8, label="AI Strategy")
    ax.plot(bench.index, bench / 1e6, color="#ff6666", lw=1.3, alpha=0.8, label="S&P 500 (SPY)")
    ax.fill_between(strat.index, strat / 1e6, bench / 1e6,
                    where=strat > bench, color="#00ff88", alpha=0.08)
    ax.fill_between(strat.index, strat / 1e6, bench / 1e6,
                    where=strat <= bench, color="#ff4444", alpha=0.08)
    ax.set_title("Equity Curve ($M)", fontsize=13, color="white")
    ax.legend(framealpha=0.3, fontsize=10)
    ax.set_ylabel("Portfolio ($M)")
    ax.grid(alpha=0.15)
    stats_text = (f"Return: {metrics['strategy_return']:.1%} vs {metrics['benchmark_return']:.1%}\n"
                  f"Sharpe: {metrics['strategy_sharpe']:.2f}  |  Sortino: {metrics['strategy_sortino']:.2f}\n"
                  f"Max DD: {metrics['strategy_max_dd']:.1%}  |  Calmar: {metrics['strategy_calmar']:.2f}")
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.6, edgecolor='#00ff88'))

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
                    ax.text(j, i, f"{v:+.1%}", ha="center", va="center", fontsize=7,
                            fontweight="bold", color="black" if abs(v) < 0.03 else "white")
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

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg_mod.CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
