"""
Day Trading Micro-Model
========================
A small, fast neural network trained on intraday candle patterns to predict:
  1. Entry timing (when to open a position)
  2. Exit timing (when to close)
  3. Take-profit levels (adaptive based on volatility)
  4. Stop-loss levels (adaptive based on volatility)

Designed to run alongside the main ensemble model as a timing overlay.
Learns candle patterns: doji, hammer, engulfing, pin bars, volume spikes,
and volatility clustering to understand intraday rhythm.

Usage:
    from src.day_model import DayTradingModel, train_day_model, generate_day_signals

Pure NumPy — no external ML dependencies.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional


class CandleFeatureExtractor:
    """Extract candle pattern features from OHLCV data for day trading."""

    @staticmethod
    def extract(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build candle-pattern features from OHLCV.
        Expects columns: Open, High, Low, Close, Volume.
        Returns a DataFrame of features aligned to the input index.
        """
        o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
        f = pd.DataFrame(index=df.index)

        body = c - o
        full_range = (h - l).replace(0, 1e-10)
        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

        # -- Candle structure --
        f["body_pct"] = body / c.shift(1).replace(0, 1e-10)
        f["range_pct"] = full_range / c.shift(1).replace(0, 1e-10)
        f["upper_wick_ratio"] = upper_wick / full_range
        f["lower_wick_ratio"] = lower_wick / full_range
        f["body_range_ratio"] = body.abs() / full_range

        # -- Pattern signals --
        # Doji: tiny body relative to range
        f["is_doji"] = (body.abs() / full_range < 0.1).astype(float)
        # Hammer: long lower wick, small body at top
        f["is_hammer"] = ((lower_wick / full_range > 0.6) & (body.abs() / full_range < 0.3)).astype(float)
        # Shooting star: long upper wick, small body at bottom
        f["is_shooting_star"] = ((upper_wick / full_range > 0.6) & (body.abs() / full_range < 0.3)).astype(float)
        # Engulfing: this body completely covers previous body
        prev_body = body.shift(1)
        f["is_bullish_engulf"] = ((body > 0) & (body.abs() > prev_body.abs()) & (prev_body < 0)).astype(float)
        f["is_bearish_engulf"] = ((body < 0) & (body.abs() > prev_body.abs()) & (prev_body > 0)).astype(float)

        # -- Volatility features --
        atr14 = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
        f["atr_pct"] = atr14 / c.replace(0, 1e-10)
        f["atr_expansion"] = atr14 / atr14.rolling(50).mean().replace(0, 1e-10)
        f["hvol_5"] = c.pct_change().rolling(5).std() * np.sqrt(252)
        f["hvol_10"] = c.pct_change().rolling(10).std() * np.sqrt(252)
        f["vol_ratio"] = f["hvol_5"] / f["hvol_10"].replace(0, 1e-10)

        # -- Momentum micro-features --
        for w in [1, 2, 3, 5]:
            f[f"ret_{w}"] = c.pct_change(w)
        f["gap_pct"] = (o - c.shift(1)) / c.shift(1).replace(0, 1e-10)

        # -- Volume features --
        vsma = v.rolling(20).mean().replace(0, 1e-10)
        f["vol_spike"] = v / vsma
        f["vol_trend"] = vsma.pct_change(5)

        # -- RSI micro (fast) --
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rs = gain / loss.replace(0, 1e-10)
        f["rsi_7"] = 100 - 100 / (1 + rs)
        f["rsi_7_norm"] = f["rsi_7"] / 100 - 0.5

        # -- Stochastic fast --
        low5 = l.rolling(5).min()
        high5 = h.rolling(5).max()
        f["stoch_fast"] = (c - low5) / (high5 - low5).replace(0, 1e-10)

        # -- Price position in range --
        f["close_in_range"] = (c - l) / full_range
        f["open_in_range"] = (o - l) / full_range

        # -- Streak features --
        up = (c > c.shift(1)).astype(float)
        down = (c < c.shift(1)).astype(float)
        f["up_streak"] = up.rolling(5).sum()
        f["down_streak"] = down.rolling(5).sum()

        # -- Time-of-bar momentum (for intraday or bar-level) --
        f["bar_momentum"] = (c - o) / full_range  # bullish/bearish bar strength

        # Clean
        f.replace([np.inf, -np.inf], np.nan, inplace=True)
        f.dropna(inplace=True)
        return f


class DayTradingMLP:
    """
    Small MLP for day trading decisions.
    3 outputs:
      [0] entry_score  (-1 to 1, positive = go long, negative = stay out)
      [1] tp_mult      (take-profit as multiple of ATR, e.g. 1.5 = 1.5×ATR)
      [2] sl_mult      (stop-loss as multiple of ATR, e.g. 1.0 = 1.0×ATR)
    """

    def __init__(self, input_dim: int, hidden: Tuple[int, ...] = (64, 32), dropout: float = 0.15, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.dropout = dropout
        self.training = True

        layer_sizes = [input_dim] + list(hidden) + [3]  # 3 outputs
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            W = self.rng.randn(fan_in, fan_out) * scale
            b = np.zeros((1, fan_out))
            self.layers.append({"W": W, "b": b})

        self.adam_m = [{"W": np.zeros_like(l["W"]), "b": np.zeros_like(l["b"])} for l in self.layers]
        self.adam_v = [{"W": np.zeros_like(l["W"]), "b": np.zeros_like(l["b"])} for l in self.layers]
        self.adam_t = 0

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.cache = {"A": [X], "Z": [], "masks": []}
        A = X
        for i, layer in enumerate(self.layers):
            Z = A @ layer["W"] + layer["b"]
            self.cache["Z"].append(Z)
            if i < len(self.layers) - 1:
                A = np.where(Z > 0, Z, 0.01 * Z)  # leaky relu
                if self.training and self.dropout > 0:
                    mask = (self.rng.rand(*A.shape) > self.dropout).astype(float)
                    A = A * mask / (1 - self.dropout)
                    self.cache["masks"].append(mask)
                else:
                    self.cache["masks"].append(np.ones_like(A))
            else:
                # Output layer: tanh for entry_score, softplus for tp/sl
                A_out = np.zeros_like(Z)
                A_out[:, 0] = np.tanh(Z[:, 0])  # entry score in [-1, 1]
                A_out[:, 1] = np.log1p(np.exp(np.clip(Z[:, 1], -10, 10))) + 0.5  # tp_mult >= 0.5
                A_out[:, 2] = np.log1p(np.exp(np.clip(Z[:, 2], -10, 10))) + 0.3  # sl_mult >= 0.3
                A = A_out
            self.cache["A"].append(A)
        return A

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, lr: float,
                 weight_decay: float = 1e-5, grad_clip: float = 1.0):
        m = len(y_true)
        # MSE gradient for all 3 outputs
        diff = y_pred - y_true
        dA = (2.0 * diff) / m

        self.adam_t += 1
        grads = []

        for i in reversed(range(len(self.layers))):
            A_prev = self.cache["A"][i]
            dW = A_prev.T @ dA + weight_decay * self.layers[i]["W"]
            db = dA.sum(axis=0, keepdims=True)
            grads.insert(0, {"W": dW, "b": db})

            if i > 0:
                dA = dA @ self.layers[i]["W"].T
                Z = self.cache["Z"][i - 1]
                dA = dA * np.where(Z > 0, 1.0, 0.01)  # leaky relu grad
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

    def predict(self, X: np.ndarray) -> np.ndarray:
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


def _build_training_targets(df: pd.DataFrame, features: pd.DataFrame, atr_series: pd.Series) -> np.ndarray:
    """
    Build training targets by simulating what WOULD have been optimal.

    For each bar, look forward to see:
      - Was going long profitable? (entry_score target)
      - What was the optimal TP in ATR multiples? (tp_mult target)
      - What SL would have avoided the worst drawdown? (sl_mult target)
    """
    close = df["Close"].reindex(features.index)
    high = df["High"].reindex(features.index)
    low = df["Low"].reindex(features.index)
    atr = atr_series.reindex(features.index)

    n = len(features)
    targets = np.zeros((n, 3))
    lookahead = 10  # bars to look forward

    for i in range(n - lookahead):
        c_now = close.iloc[i]
        atr_now = atr.iloc[i]
        if c_now <= 0 or atr_now <= 0:
            continue

        future_close = close.iloc[i + 1:i + 1 + lookahead]
        future_high = high.iloc[i + 1:i + 1 + lookahead]
        future_low = low.iloc[i + 1:i + 1 + lookahead]

        if len(future_close) < lookahead:
            continue

        # Max gain and max adverse
        max_gain = (future_high.max() - c_now) / c_now
        max_loss = (c_now - future_low.min()) / c_now
        net_return = (future_close.iloc[-1] - c_now) / c_now

        # Entry score: blend of net return direction and risk/reward
        rr_ratio = max_gain / max(max_loss, 1e-6)
        entry_signal = np.tanh(net_return * 20)  # scale to [-1, 1]
        # Boost if risk/reward was favorable
        if rr_ratio > 2.0:
            entry_signal = np.clip(entry_signal + 0.3, -1, 1)
        elif rr_ratio < 0.5:
            entry_signal = np.clip(entry_signal - 0.3, -1, 1)
        targets[i, 0] = entry_signal

        # Optimal TP: ATR multiple of max gain
        targets[i, 1] = np.clip(max_gain / (atr_now / c_now), 0.5, 5.0)

        # Optimal SL: ATR multiple that would have contained loss
        targets[i, 2] = np.clip(max_loss / (atr_now / c_now) * 0.8, 0.3, 3.0)

    # Trim last lookahead rows (no target available)
    return targets[:n - lookahead]


def train_day_model(
    ohlcv_data: Dict[str, pd.DataFrame],
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 5e-4,
    hidden: Tuple[int, ...] = (64, 32),
    val_split: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Train the day trading micro-model on historical candle data.

    Args:
        ohlcv_data: dict of ticker -> DataFrame with OHLCV columns
        epochs: training epochs
        batch_size: mini-batch size
        lr: learning rate
        hidden: hidden layer sizes
        val_split: fraction held out for validation
        seed: random seed

    Returns:
        dict with 'model', 'norm_stats', 'metrics'
    """
    extractor = CandleFeatureExtractor()
    all_X, all_y = [], []

    for ticker, df in ohlcv_data.items():
        if len(df) < 100:
            continue

        features = extractor.extract(df)
        if features.empty:
            continue

        # ATR for target computation
        h, l, c = df["High"], df["Low"], df["Close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        targets = _build_training_targets(df, features, atr)
        feat_values = features.values[:len(targets)]

        # Remove NaN rows
        mask = ~(np.isnan(feat_values).any(axis=1) | np.isnan(targets).any(axis=1))
        all_X.append(feat_values[mask])
        all_y.append(targets[mask])

    if not all_X:
        raise ValueError("No valid training data")

    X = np.concatenate(all_X, axis=0).astype(np.float64)
    y = np.concatenate(all_y, axis=0).astype(np.float64)

    # Normalize features
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-10] = 1.0
    X = (X - mu) / sigma

    # Remove any remaining NaN/Inf
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1) | np.isinf(X).any(axis=1))
    X, y = X[valid], y[valid]

    # Temporal split
    split = int(len(X) * (1 - val_split))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    input_dim = X.shape[1]
    model = DayTradingMLP(input_dim, hidden=hidden, seed=seed)

    best_val_loss = float("inf")
    best_params = None
    patience = 10
    patience_ctr = 0
    current_lr = lr

    print(f"  Training day model: {X_tr.shape[0]:,} train, {X_val.shape[0]:,} val, {input_dim} features")

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(len(X_tr))
        X_shuf, y_shuf = X_tr[perm], y_tr[perm]

        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(X_tr), batch_size):
            end = min(start + batch_size, len(X_tr))
            xb, yb = X_shuf[start:end], y_shuf[start:end]
            pred = model.forward(xb)
            loss = np.mean((pred - yb) ** 2)
            epoch_loss += loss
            batches += 1
            model.backward(pred, yb, current_lr)

        # Validation
        val_pred = model.predict(X_val)
        val_loss = np.mean((val_pred - y_val) ** 2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = model.get_params()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"    Early stop at epoch {epoch}, best val loss: {best_val_loss:.6f}")
                break

        current_lr *= 0.995
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: train={epoch_loss/batches:.6f} val={val_loss:.6f}")

    model.set_params(best_params)

    # Evaluate on validation set
    val_pred = model.predict(X_val)
    entry_corr = np.corrcoef(val_pred[:, 0], y_val[:, 0])[0, 1] if val_pred[:, 0].std() > 0 else 0
    entry_dir_acc = np.mean((val_pred[:, 0] > 0) == (y_val[:, 0] > 0))

    metrics = {
        "val_loss": float(best_val_loss),
        "entry_correlation": float(entry_corr),
        "entry_direction_accuracy": float(entry_dir_acc),
        "train_samples": len(X_tr),
        "val_samples": len(X_val),
        "params": model.param_count(),
    }

    print(f"  Day model trained: entry_corr={entry_corr:.4f}, dir_acc={entry_dir_acc:.2%}, params={model.param_count():,}")

    return {
        "model": model,
        "norm_stats": {"mu": mu, "sigma": sigma},
        "metrics": metrics,
        "input_dim": input_dim,
        "hidden": hidden,
    }


def generate_day_signals(
    model_bundle: Dict,
    ohlcv_df: pd.DataFrame,
) -> Dict:
    """
    Run the day model on current candle data to get entry/exit/TP/SL signals.

    Returns:
        dict with:
          entry_score: float in [-1, 1] (>0.3 = strong buy, <-0.3 = stay out)
          tp_atr_mult: suggested take-profit in ATR multiples
          sl_atr_mult: suggested stop-loss in ATR multiples
          tp_pct: take-profit as price percentage
          sl_pct: stop-loss as price percentage
          should_enter: bool
          should_exit: bool (if entry_score strongly negative while in position)
    """
    model = model_bundle["model"]
    mu = model_bundle["norm_stats"]["mu"]
    sigma = model_bundle["norm_stats"]["sigma"]

    extractor = CandleFeatureExtractor()
    features = extractor.extract(ohlcv_df)

    if features.empty:
        return {"entry_score": 0.0, "tp_atr_mult": 2.0, "sl_atr_mult": 1.0,
                "tp_pct": 0.03, "sl_pct": 0.015, "should_enter": False, "should_exit": False}

    x = features.iloc[-1:].values.astype(np.float64)
    x = (x - mu) / sigma
    x = np.nan_to_num(x, 0.0)

    pred = model.predict(x)[0]
    entry_score = float(pred[0])
    tp_mult = float(pred[1])
    sl_mult = float(pred[2])

    # Get current ATR for converting to percentages
    h, l, c = ohlcv_df["High"], ohlcv_df["Low"], ohlcv_df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    current_price = float(c.iloc[-1])
    atr_pct = atr / current_price if current_price > 0 else 0.02

    tp_pct = tp_mult * atr_pct
    sl_pct = sl_mult * atr_pct

    # Clamp to reasonable ranges
    tp_pct = np.clip(tp_pct, 0.005, 0.15)
    sl_pct = np.clip(sl_pct, 0.003, 0.08)

    should_enter = entry_score > 0.3
    should_exit = entry_score < -0.3

    return {
        "entry_score": round(entry_score, 4),
        "tp_atr_mult": round(tp_mult, 2),
        "sl_atr_mult": round(sl_mult, 2),
        "tp_pct": round(float(tp_pct), 4),
        "sl_pct": round(float(sl_pct), 4),
        "should_enter": bool(should_enter),
        "should_exit": bool(should_exit),
        "atr_pct": round(float(atr_pct), 4),
    }


def save_day_model(bundle: Dict, path: Path):
    """Persist the day trading model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "params": bundle["model"].get_params(),
        "norm_stats": bundle["norm_stats"],
        "metrics": bundle["metrics"],
        "input_dim": bundle["input_dim"],
        "hidden": bundle["hidden"],
    }
    with path.open("wb") as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_day_model(path: Path) -> Dict:
    """Load a persisted day trading model."""
    path = Path(path)
    with path.open("rb") as f:
        data = pickle.load(f)
    model = DayTradingMLP(data["input_dim"], hidden=data["hidden"])
    model.set_params(data["params"])
    return {
        "model": model,
        "norm_stats": data["norm_stats"],
        "metrics": data["metrics"],
        "input_dim": data["input_dim"],
        "hidden": data["hidden"],
    }
