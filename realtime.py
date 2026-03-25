#!/usr/bin/env python3
"""
Realtime trader that uses trained models from main.py.

Features:
  - Loads persisted ensemble bundle (outputs/model_bundle.pkl)
  - Pulls live/delayed bars from Yahoo Finance
  - Smart signal filtering with ensemble agreement + volatility scaling
  - Per-position stop-loss, trailing stop, and take-profit
  - Portfolio-level drawdown circuit breaker
  - Signal smoothing across cycles to reduce whipsaw
  - Volatility-adjusted position sizing
  - Optional Alpaca execution (paper or live)
  - Persistent state across restarts
  - Small local HTTP API for online usage
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta, time as dt_time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib import request as urlrequest
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config as cfg
from utils import (
    add_features,
    apply_normalisation,
    load_model_bundle,
    print_header,
    print_metric,
    print_section,
    colour,
)
from utils import _signal_to_weights


# ═══════════════════════════════════════════════════════════════════════════════
#  PERSISTENT TRADING STATE
# ═══════════════════════════════════════════════════════════════════════════════

class TradingState:
    """Tracks positions, P&L, drawdowns, and signal history across cycles."""

    def __init__(self, state_path: Path = None):
        self.state_path = state_path or cfg.RT_STATE_PATH
        self.positions: Dict[str, Dict[str, float]] = {}  # sym -> {entry_price, peak_price, shares, entry_time}
        self.signal_history: Dict[str, List[float]] = {}   # sym -> list of recent signals
        self.equity_peak: float = 0.0
        self.last_drawdown_halt: Optional[str] = None
        self.cycle_count: int = 0
        self.total_realized_pnl: float = 0.0
        self._load()

    def _load(self):
        if self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text())
                self.positions = raw.get("positions", {})
                self.signal_history = raw.get("signal_history", {})
                self.equity_peak = raw.get("equity_peak", 0.0)
                self.last_drawdown_halt = raw.get("last_drawdown_halt")
                self.cycle_count = raw.get("cycle_count", 0)
                self.total_realized_pnl = raw.get("total_realized_pnl", 0.0)
            except (json.JSONDecodeError, KeyError):
                pass

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "positions": self.positions,
            "signal_history": self.signal_history,
            "equity_peak": self.equity_peak,
            "last_drawdown_halt": self.last_drawdown_halt,
            "cycle_count": self.cycle_count,
            "total_realized_pnl": self.total_realized_pnl,
            "updated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
        }
        self.state_path.write_text(json.dumps(data, indent=2))

    def update_signal_history(self, signals: Dict[str, float], max_len: int = 10):
        for sym, sig in signals.items():
            if sym not in self.signal_history:
                self.signal_history[sym] = []
            self.signal_history[sym].append(sig)
            if len(self.signal_history[sym]) > max_len:
                self.signal_history[sym] = self.signal_history[sym][-max_len:]

    def get_smoothed_signal(self, sym: str, span: int = 3) -> float:
        hist = self.signal_history.get(sym, [])
        if not hist:
            return 0.0
        if len(hist) == 1:
            return hist[0]
        weights = np.array([np.exp(-i / max(span, 1)) for i in range(len(hist) - 1, -1, -1)])
        weights /= weights.sum()
        return float(np.dot(weights, hist))

    def record_position(self, sym: str, entry_price: float, shares: float):
        self.positions[sym] = {
            "entry_price": entry_price,
            "peak_price": entry_price,
            "shares": shares,
            "entry_time": datetime.now(ZoneInfo("UTC")).isoformat(),
        }

    def update_peak_price(self, sym: str, current_price: float):
        if sym in self.positions:
            if current_price > self.positions[sym].get("peak_price", 0):
                self.positions[sym]["peak_price"] = current_price

    def close_position(self, sym: str, exit_price: float) -> float:
        if sym not in self.positions:
            return 0.0
        pos = self.positions.pop(sym)
        pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        self.total_realized_pnl += pnl
        return pnl

    def is_drawdown_halted(self, current_equity: float) -> bool:
        if self.equity_peak > 0:
            dd = (self.equity_peak - current_equity) / self.equity_peak
            if dd >= cfg.RT_MAX_DRAWDOWN_HALT:
                self.last_drawdown_halt = datetime.now(ZoneInfo("UTC")).isoformat()
                return True
        if self.last_drawdown_halt:
            halt_time = datetime.fromisoformat(self.last_drawdown_halt)
            cooldown = timedelta(hours=cfg.RT_DRAWDOWN_COOLDOWN_HOURS)
            if datetime.now(ZoneInfo("UTC")) - halt_time < cooldown:
                return True
            self.last_drawdown_halt = None
        return False

    def update_equity_peak(self, equity: float):
        if equity > self.equity_peak:
            self.equity_peak = equity


# ═══════════════════════════════════════════════════════════════════════════════
#  ENV + DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_env_file(env_path: Path):
    """Load KEY=VALUE pairs from a local .env file into process env."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def fetch_live_ohlcv(tickers, period=None, interval="1d", start_date=None, end_date=None):
    """Fetch OHLCV bars for all tickers via yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance for realtime mode: pip install yfinance") from exc

    data = {}
    for t in tickers:
        kwargs = {
            "interval": interval,
            "auto_adjust": True,
            "progress": False,
            "threads": False,
        }
        if start_date or end_date:
            kwargs["start"] = start_date
            kwargs["end"] = end_date
        else:
            kwargs["period"] = period or "60d"
        df = yf.download(t, **kwargs)
        if df.empty:
            raise RuntimeError(f"No live bars returned for {t}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in df.columns:
                raise RuntimeError(f"Missing column {c} for {t}")
        data[t] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  SMART SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_market_regime(live_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Detect market regime from live data to adjust trading behavior."""
    all_returns = []
    for t, df in live_data.items():
        rets = df["Close"].pct_change().dropna().values
        if len(rets) > 5:
            all_returns.append(rets)

    if not all_returns:
        return {"regime": "normal", "vol_scale": 1.0, "trend": 0.0}

    # Use the average cross-sectional behavior
    min_len = min(len(r) for r in all_returns)
    stacked = np.column_stack([r[-min_len:] for r in all_returns])
    avg_rets = stacked.mean(axis=1)

    # Recent vs longer-term volatility
    recent_vol = np.std(avg_rets[-10:]) * np.sqrt(252) if len(avg_rets) >= 10 else 0.15
    longer_vol = np.std(avg_rets[-60:]) * np.sqrt(252) if len(avg_rets) >= 60 else 0.15

    # Trend: average return over last 20 days annualized
    trend_20 = np.mean(avg_rets[-20:]) * 252 if len(avg_rets) >= 20 else 0.0

    # Regime classification
    vol_ratio = recent_vol / max(longer_vol, 0.01)
    if vol_ratio > 1.5:
        regime = "high_vol"
    elif vol_ratio < 0.7 and trend_20 > 0.05:
        regime = "low_vol_bull"
    elif trend_20 < -0.10:
        regime = "bearish"
    elif trend_20 > 0.10:
        regime = "bullish"
    else:
        regime = "normal"

    # Volatility scaling: reduce exposure in high vol, increase in low vol
    target_vol = cfg.RT_VOL_TARGET
    vol_scale = target_vol / max(recent_vol, 0.01)
    vol_scale = np.clip(vol_scale, 0.3, 1.5)

    return {
        "regime": regime,
        "vol_scale": float(vol_scale),
        "recent_vol": float(recent_vol),
        "longer_vol": float(longer_vol),
        "trend_20d": float(trend_20),
        "vol_ratio": float(vol_ratio),
    }


def compute_per_ticker_volatility(live_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Compute annualized volatility per ticker for position sizing."""
    vols = {}
    lookback = cfg.RT_VOL_LOOKBACK_DAYS
    for t, df in live_data.items():
        rets = df["Close"].pct_change().dropna()
        if len(rets) >= lookback:
            vols[t] = float(rets.iloc[-lookback:].std() * np.sqrt(252))
        elif len(rets) >= 5:
            vols[t] = float(rets.std() * np.sqrt(252))
        else:
            vols[t] = 0.20  # default 20% vol
    return vols


def score_entry_timing(feat_raw: pd.DataFrame, signal_direction: float) -> Dict[str, Any]:
    """
    Score whether NOW is a good time to enter a position, using the raw
    technical features already computed by add_features().

    Returns a dict with overall score (0-1), component scores, and whether
    entry is approved.
    """
    if feat_raw.empty:
        return {"score": 0.0, "approved": False, "reason": "no_data"}

    latest = feat_raw.iloc[-1]
    is_long = signal_direction > 0
    components = {}

    # 1) RSI timing — don't buy overbought, prefer oversold for longs
    rsi = latest.get("rsi", 50)
    if is_long:
        if rsi > cfg.RT_RSI_OVERBOUGHT:
            components["rsi"] = 0.0  # terrible entry — overbought
        elif rsi < cfg.RT_RSI_OVERSOLD:
            components["rsi"] = 1.0  # perfect — oversold bounce
        else:
            # Linear scale: oversold=1.0 to overbought=0.0
            components["rsi"] = max(0, (cfg.RT_RSI_OVERBOUGHT - rsi) / (cfg.RT_RSI_OVERBOUGHT - cfg.RT_RSI_OVERSOLD))
    else:
        # For shorts (rare, but handle): prefer overbought
        components["rsi"] = max(0, min(1, (rsi - 50) / 30))

    # 2) Bollinger Band position — don't buy at top of band
    bb_pct = latest.get("bb_pct", 0.5)
    if is_long:
        if bb_pct > cfg.RT_BB_UPPER_AVOID:
            components["bb"] = 0.1  # near upper band — bad long entry
        elif bb_pct < 0.3:
            components["bb"] = 1.0  # near lower band — great long entry
        else:
            components["bb"] = max(0, 1.0 - bb_pct)
    else:
        components["bb"] = max(0, bb_pct)  # for shorts, high bb_pct is good

    # 3) MACD confirmation — histogram should agree with signal direction
    macd_hist = latest.get("macd_hist", 0)
    if cfg.RT_MACD_CONFIRM_REQUIRED:
        if (is_long and macd_hist > 0) or (not is_long and macd_hist < 0):
            components["macd"] = 1.0
        elif abs(macd_hist) < 0.001:
            components["macd"] = 0.5  # neutral, not great
        else:
            components["macd"] = 0.2  # MACD disagrees with signal
    else:
        components["macd"] = 0.7  # default if not required

    # 4) Volume confirmation — need above-average volume for conviction
    vol_ratio = latest.get("vol_ratio", 1.0)
    if vol_ratio >= cfg.RT_VOLUME_CONFIRM_MIN:
        components["volume"] = min(1.0, vol_ratio / 1.5)
    else:
        components["volume"] = vol_ratio / cfg.RT_VOLUME_CONFIRM_MIN * 0.5

    # 5) Momentum alignment — is recent momentum in our direction?
    mom = latest.get("mom_composite", 0)
    if (is_long and mom > 0) or (not is_long and mom < 0):
        components["momentum"] = min(1.0, abs(mom) * 20 + 0.5)
    else:
        components["momentum"] = max(0, 0.5 - abs(mom) * 10)

    # 6) Mean reversion signal — for longs, prefer prices below moving avg
    mean_rev = latest.get("mean_rev_20", 0)
    if is_long:
        # mean_rev_20 is -(price/ma - 1), so positive means price < MA
        components["mean_reversion"] = min(1.0, max(0, 0.5 + mean_rev * 5))
    else:
        components["mean_reversion"] = min(1.0, max(0, 0.5 - mean_rev * 5))

    # 7) Stochastic — confirm not overbought/oversold against our direction
    stoch_k = latest.get("stoch_k", 0.5)
    if is_long:
        components["stochastic"] = max(0, 1.0 - stoch_k)  # low stoch = good long entry
    else:
        components["stochastic"] = max(0, stoch_k)

    # Weighted average (RSI and BB matter most for timing)
    weights = {
        "rsi": 0.20, "bb": 0.18, "macd": 0.18, "volume": 0.12,
        "momentum": 0.12, "mean_reversion": 0.10, "stochastic": 0.10,
    }
    score = sum(components[k] * weights[k] for k in components)
    approved = score >= cfg.RT_ENTRY_SCORE_MIN

    reason = "approved" if approved else "poor_timing"
    # Override: if RSI is extremely overbought and we're going long, block it
    if is_long and rsi > cfg.RT_RSI_OVERBOUGHT:
        approved = False
        reason = "rsi_overbought"

    return {
        "score": round(score, 3),
        "approved": approved,
        "reason": reason,
        "components": {k: round(v, 3) for k, v in components.items()},
    }


def ai_review_positions(
    bundle: Dict[str, Any],
    live_data: Dict[str, pd.DataFrame],
    state: TradingState,
    held_symbols: List[str],
) -> Dict[str, Dict]:
    """
    Re-run the AI model on every held position and decide whether to keep or exit.
    Also ranks positions by expected return to reallocate capital.
    """
    settings = bundle["settings"]
    lookback = int(settings["lookback_window"])
    norm_stats = bundle["norm_stats"]
    models = bundle["models"]
    ensemble_weights = np.array(bundle["ensemble_weights"], dtype=np.float64)

    def default_hold_review(reason: str) -> Dict[str, Any]:
        return {
            "action": "hold",
            "reason": reason,
            "prediction": 0.0,
            "hold_score": 0.0,
            "bearish_models": 0,
            "total_models": len(models),
            "consecutive_decay": 0,
            "unrealized_pct": 0.0,
        }

    reviews = {}
    for sym in held_symbols:
        if sym not in live_data or sym not in norm_stats:
            reviews[sym] = default_hold_review("no_data")
            continue

        df = live_data[sym]
        feat_raw = add_features(df["Close"], df["High"], df["Low"], df["Volume"])
        mu, sigma = norm_stats[sym]
        feat_norm = apply_normalisation(feat_raw, mu, sigma)

        if len(feat_norm) < lookback:
            reviews[sym] = default_hold_review("insufficient_data")
            continue

        x = feat_norm.iloc[-lookback:].values.flatten().astype(np.float64).reshape(1, -1)

        # Get current model prediction
        per_model_preds = [float(m.predict(x)[0]) for m in models]
        current_pred = sum(w * p for w, p in zip(ensemble_weights, per_model_preds))

        # Check signal history for decay pattern
        sig_hist = state.signal_history.get(sym, [])
        consecutive_decay = 0
        if len(sig_hist) >= 2:
            for i in range(len(sig_hist) - 1, 0, -1):
                if abs(sig_hist[i]) < abs(sig_hist[i - 1]):
                    consecutive_decay += 1
                else:
                    break

        # Check if model has flipped against the position
        pos_info = state.positions.get(sym, {})
        entry_price = pos_info.get("entry_price", 0)
        current_price = float(df["Close"].iloc[-1])
        is_long = True  # we only go long in this system
        model_flipped = is_long and current_pred < cfg.RT_AI_EXIT_SIGNAL_FLIP

        # Check ensemble agreement on current direction
        n_bearish = sum(1 for p in per_model_preds if p < 0)
        majority_bearish = n_bearish > len(per_model_preds) / 2

        action = "hold"
        reason = "model_bullish"

        # Decision logic: exit if AI says so
        if model_flipped and majority_bearish:
            action = "ai_exit"
            reason = f"model_flipped (pred={current_pred:+.4f}, {n_bearish}/{len(per_model_preds)} bearish)"
        elif consecutive_decay >= cfg.RT_AI_EXIT_DECAY_CYCLES:
            action = "ai_exit"
            reason = f"signal_decay ({consecutive_decay} cycles declining)"
        elif model_flipped:
            action = "ai_reduce"  # model negative but not all agree — reduce, don't exit
            reason = f"model_negative (pred={current_pred:+.4f}, reducing)"
        elif majority_bearish and current_pred < 0:
            action = "ai_reduce"
            reason = f"majority_bearish ({n_bearish}/{len(per_model_preds)} models)"

        # Compute "hold quality" score for ranking positions
        hold_score = current_pred  # higher = more worth holding

        reviews[sym] = {
            "action": action,
            "reason": reason,
            "prediction": round(current_pred, 6),
            "hold_score": round(hold_score, 6),
            "bearish_models": n_bearish,
            "total_models": len(per_model_preds),
            "consecutive_decay": consecutive_decay,
            "unrealized_pct": round((current_price - entry_price) / entry_price, 4) if entry_price > 0 else 0,
        }

    return reviews


def build_live_signals(
    bundle: Dict[str, Any],
    period: str,
    interval: str,
    state: TradingState = None,
    start_date: str = None,
    end_date: str = None,
):
    """Generate signals with ensemble agreement, smoothing, regime awareness, and entry timing."""
    settings = bundle["settings"]
    tickers = settings["universe"]
    lookback = int(settings["lookback_window"])
    norm_stats = bundle["norm_stats"]
    models = bundle["models"]
    ensemble_weights = np.array(bundle["ensemble_weights"], dtype=np.float64)

    live = fetch_live_ohlcv(
        tickers,
        period=period,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
    )

    # Market regime detection
    regime_info = compute_market_regime(live)
    ticker_vols = compute_per_ticker_volatility(live)

    raw_preds = {}
    model_directions = {}  # sym -> list of +1/-1 per model
    latest_ts = None
    latest_prices = {}
    raw_features = {}      # sym -> raw feature DataFrame (for entry timing)

    for t in tickers:
        df = live[t]
        feat_raw = add_features(df["Close"], df["High"], df["Low"], df["Volume"])
        raw_features[t] = feat_raw
        if t not in norm_stats:
            raise RuntimeError(f"Ticker {t} missing normalisation stats in bundle.")
        mu, sigma = norm_stats[t]
        feat_norm = apply_normalisation(feat_raw, mu, sigma)
        if len(feat_norm) < lookback:
            raise RuntimeError(f"Not enough bars for {t}: have {len(feat_norm)} need {lookback}")

        x = feat_norm.iloc[-lookback:].values.flatten().astype(np.float64).reshape(1, -1)

        # Collect individual model predictions for agreement check
        per_model_preds = []
        for m in models:
            per_model_preds.append(float(m.predict(x)[0]))

        weighted_pred = sum(w * p for w, p in zip(ensemble_weights, per_model_preds))
        raw_preds[t] = weighted_pred

        # Track model direction agreement
        directions = [1 if p > 0 else -1 for p in per_model_preds]
        model_directions[t] = directions
        latest_ts = feat_norm.index[-1]
        latest_prices[t] = float(df["Close"].iloc[-1])

    # Update signal history for smoothing
    if state is not None:
        state.update_signal_history(raw_preds)

    # Apply signal filtering + entry timing
    filtered_signals = {}
    signal_meta = {}
    entry_scores = {}
    for t in tickers:
        raw_sig = raw_preds[t]

        # 1) Signal smoothing across cycles (reduces whipsaw)
        if state is not None and cfg.RT_SIGNAL_SMOOTHING > 1:
            smoothed = state.get_smoothed_signal(t, span=cfg.RT_SIGNAL_SMOOTHING)
        else:
            smoothed = raw_sig

        # 2) Ensemble agreement filter
        dirs = model_directions[t]
        n_agree = sum(1 for d in dirs if d == (1 if smoothed > 0 else -1))
        agreement = n_agree / len(dirs)

        # 3) Minimum signal strength filter
        if abs(smoothed) < cfg.RT_MIN_SIGNAL_STRENGTH:
            filtered_signals[t] = 0.0
            entry_scores[t] = {"score": 0, "approved": False, "reason": "signal_too_weak"}
            signal_meta[t] = {"raw": raw_sig, "smoothed": smoothed, "agreement": agreement, "action": "filtered_weak"}
            continue

        # 4) Agreement filter: require majority of models to agree
        if agreement < cfg.RT_ENSEMBLE_AGREEMENT_MIN:
            filtered_signals[t] = smoothed * 0.3
            entry_scores[t] = {"score": 0, "approved": False, "reason": "ensemble_disagrees"}
            signal_meta[t] = {"raw": raw_sig, "smoothed": smoothed, "agreement": agreement, "action": "reduced_disagreement"}
            continue

        # 5) Entry timing: check if technicals support entering NOW
        entry_timing = score_entry_timing(raw_features[t], smoothed)
        entry_scores[t] = entry_timing

        # If entry timing is bad, reduce the signal (don't zero it — we might
        # already hold this position and want to keep it, just not add)
        is_new_position = state is None or t not in state.positions
        if is_new_position and not entry_timing["approved"]:
            filtered_signals[t] = smoothed * 0.15  # drastically reduce for new entries with bad timing
            signal_meta[t] = {
                "raw": raw_sig, "smoothed": smoothed, "agreement": agreement,
                "entry_score": entry_timing["score"], "entry_reason": entry_timing["reason"],
                "action": "blocked_bad_entry",
            }
            continue

        # 6) Volatility scaling: scale down in high-vol names, up in low-vol
        vol_adj = 1.0
        if cfg.RT_VOL_SCALE_ENABLED:
            ticker_vol = ticker_vols.get(t, 0.20)
            vol_adj = cfg.RT_VOL_TARGET / max(ticker_vol, 0.01)
            vol_adj = np.clip(vol_adj, 0.3, 2.0)

        # 7) Regime adjustment
        regime_adj = 1.0
        regime = regime_info["regime"]
        if regime == "bearish":
            regime_adj = 0.5 if smoothed > 0 else 1.2
        elif regime == "high_vol":
            regime_adj = 0.6
        elif regime == "low_vol_bull":
            regime_adj = 1.2 if smoothed > 0 else 0.8

        # 8) Conviction scaling: boost strong signals
        conviction = 1.0
        if abs(smoothed) > cfg.RT_STRONG_SIGNAL_THRESHOLD * 2:
            conviction = 1.5
        elif abs(smoothed) > cfg.RT_STRONG_SIGNAL_THRESHOLD:
            conviction = 1.3

        # 9) Entry timing bonus/penalty — good timing boosts signal, bad reduces it
        timing_mult = 0.7 + 0.6 * entry_timing["score"]  # range: 0.7x to 1.3x

        final_signal = smoothed * vol_adj * regime_adj * conviction * timing_mult
        filtered_signals[t] = final_signal
        signal_meta[t] = {
            "raw": raw_sig,
            "smoothed": smoothed,
            "agreement": agreement,
            "vol_adj": vol_adj,
            "regime_adj": regime_adj,
            "conviction": conviction,
            "entry_score": entry_timing["score"],
            "entry_reason": entry_timing["reason"],
            "timing_mult": round(timing_mult, 3),
            "action": "active",
        }

    signal_series = pd.Series(filtered_signals).sort_index()

    # Apply cash buffer: reduce total weight to leave cash reserve
    weights = _signal_to_weights(signal_series, list(signal_series.index), cfg)
    cash_pct = cfg.RT_CASH_BUFFER_PCT
    if cash_pct > 0:
        scale_factor = 1.0 - cash_pct
        weights = {t: w * scale_factor for t, w in weights.items()}

    # Apply portfolio-level vol scaling
    portfolio_vol_scale = regime_info["vol_scale"]
    if portfolio_vol_scale < 1.0:
        weights = {t: w * portfolio_vol_scale for t, w in weights.items()}

    return {
        "timestamp": str(latest_ts),
        "signals": {k: float(v) for k, v in signal_series.to_dict().items()},
        "raw_signals": {k: float(v) for k, v in raw_preds.items()},
        "target_weights": {k: float(v) for k, v in weights.items()},
        "latest_prices": latest_prices,
        "regime": regime_info,
        "signal_meta": signal_meta,
        "entry_scores": entry_scores,
        "ticker_vols": ticker_vols,
        "live_data": live,            # pass through for AI review
        "raw_features": raw_features,  # pass through for AI review
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ALPACA CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class AlpacaClient:
    """Small Alpaca REST wrapper for paper/live execution."""

    def __init__(self, mode: str):
        if mode not in {"paper", "live"}:
            raise ValueError("Alpaca mode must be 'paper' or 'live'")
        key = os.getenv("ALPACA_API_KEY_ID")
        secret = os.getenv("ALPACA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError(
                "Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY for broker mode "
                "(export env vars or place them in .env)."
            )
        base = "https://paper-api.alpaca.markets" if mode == "paper" else "https://api.alpaca.markets"
        self.base = base.rstrip("/")
        self.headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, payload: Dict[str, Any] = None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            f"{self.base}{path}",
            data=body,
            headers=self.headers,
            method=method.upper(),
        )
        try:
            with urlrequest.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8")
                if not raw.strip():
                    return {}
                return json.loads(raw)
        except HTTPError as exc:
            raw = ""
            try:
                raw = exc.read().decode("utf-8")
            except Exception:
                raw = ""
            msg = f"HTTP {exc.code} {exc.reason}"
            if raw:
                try:
                    err = json.loads(raw)
                    if isinstance(err, dict):
                        msg = err.get("message") or err.get("error") or msg
                except Exception:
                    msg = f"{msg} | {raw.strip()}"
            raise RuntimeError(f"Alpaca API request failed ({method.upper()} {path}): {msg}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Alpaca API request failed ({method.upper()} {path}): {exc.reason}"
            ) from exc

    def get_account(self):
        return self._request("GET", "/v2/account")

    def get_positions(self):
        return self._request("GET", "/v2/positions")

    def get_calendar(self, start_date: str, end_date: str):
        query = urlencode({"start": start_date, "end": end_date})
        return self._request("GET", f"/v2/calendar?{query}")

    def close_position(self, symbol: str):
        return self._request("DELETE", f"/v2/positions/{symbol}")

    def submit_order(self, symbol: str, notional_usd: float, side: str):
        if notional_usd < 1.0:
            return None
        payload = {
            "symbol": symbol,
            "notional": round(float(notional_usd), 2),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        return self._request("POST", "/v2/orders", payload=payload)


# ═══════════════════════════════════════════════════════════════════════════════
#  RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def check_position_risk(
    client: AlpacaClient,
    state: TradingState,
    latest_prices: Dict[str, float],
) -> Dict[str, List[Dict]]:
    """Check all positions for stop-loss, trailing stop, and take-profit triggers."""
    positions_raw = client.get_positions()
    exits = []
    updates = []

    for pos in positions_raw:
        sym = pos["symbol"]
        current_price = float(pos.get("current_price", 0))
        if current_price <= 0:
            current_price = latest_prices.get(sym, 0)
        if current_price <= 0:
            continue

        avg_entry = float(pos.get("avg_entry_price", 0))
        if avg_entry <= 0:
            continue

        unrealized_pct = (current_price - avg_entry) / avg_entry
        market_value = abs(float(pos.get("market_value", 0)))

        # Sync state with broker position
        if sym not in state.positions:
            state.record_position(sym, avg_entry, float(pos.get("qty", 0)))
        state.update_peak_price(sym, current_price)

        tracked = state.positions.get(sym, {})
        peak_price = tracked.get("peak_price", avg_entry)
        drawdown_from_peak = (peak_price - current_price) / peak_price if peak_price > 0 else 0

        reason = None

        # 1) Hard stop-loss
        if unrealized_pct <= -cfg.RT_STOP_LOSS_PCT:
            reason = f"stop_loss ({unrealized_pct:+.2%} vs -{cfg.RT_STOP_LOSS_PCT:.0%})"

        # 2) Take-profit
        elif unrealized_pct >= cfg.RT_TAKE_PROFIT_PCT:
            reason = f"take_profit ({unrealized_pct:+.2%} vs +{cfg.RT_TAKE_PROFIT_PCT:.0%})"

        # 3) Trailing stop (only if position has gained enough to activate)
        elif unrealized_pct >= cfg.RT_TRAILING_STOP_ACTIVATE and drawdown_from_peak >= cfg.RT_TRAILING_STOP_PCT:
            reason = f"trailing_stop (peak {peak_price:.2f}, drop {drawdown_from_peak:.2%})"

        if reason:
            exits.append({
                "symbol": sym,
                "reason": reason,
                "entry_price": avg_entry,
                "current_price": current_price,
                "peak_price": peak_price,
                "unrealized_pct": unrealized_pct,
                "market_value": market_value,
            })
        else:
            updates.append({
                "symbol": sym,
                "unrealized_pct": unrealized_pct,
                "peak_price": peak_price,
            })

    return {"exits": exits, "updates": updates}


def execute_risk_exits(
    client: AlpacaClient,
    state: TradingState,
    exits: List[Dict],
) -> List[Dict]:
    """Close positions that triggered risk exits."""
    results = []
    for exit_info in exits:
        sym = exit_info["symbol"]
        try:
            order = client.close_position(sym)
            pnl = state.close_position(sym, exit_info["current_price"])
            results.append({
                "symbol": sym,
                "reason": exit_info["reason"],
                "pnl": pnl,
                "status": "closed",
            })
        except Exception as exc:
            results.append({
                "symbol": sym,
                "reason": exit_info["reason"],
                "status": "failed",
                "error": str(exc),
            })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  REBALANCING
# ═══════════════════════════════════════════════════════════════════════════════

def _position_market_value(position: Dict[str, Any]) -> float:
    """Return signed market value so short positions are treated as negative."""
    market_value = float(position.get("market_value", 0.0))
    side = str(position.get("side", "")).lower()
    if side == "short":
        return -abs(market_value)
    return abs(market_value)


def rebalance_broker(
    client: AlpacaClient,
    target_weights: Dict[str, float],
    state: TradingState = None,
):
    """Rebalance against live Alpaca equity + positions with smart ordering."""
    account = client.get_account()
    equity = float(account.get("equity", 0.0))
    positions_raw = client.get_positions()
    current = {p["symbol"]: p for p in positions_raw}
    rebalance_floor = max(50.0, 0.002 * max(equity, 1.0))
    managed_symbols = {sym for sym, weight in target_weights.items() if float(weight) > 0.0}

    orders = []
    failures = []
    actions = []

    # Close positions the model is not managing
    for sym, position in current.items():
        if sym in managed_symbols:
            continue
        current_mv = _position_market_value(position)
        if abs(current_mv) < 1.0:
            continue
        try:
            order = client.close_position(sym)
            orders.append(order)
            if state:
                exit_price = float(position.get("current_price", 0))
                state.close_position(sym, exit_price)
            actions.append({
                "symbol": sym,
                "action": "close",
                "current_market_value": round(current_mv, 2),
                "target_market_value": 0.0,
                "delta_market_value": round(-current_mv, 2),
            })
        except Exception as exc:
            failures.append({
                "symbol": sym,
                "action": "close",
                "current_market_value": round(current_mv, 2),
                "target_market_value": 0.0,
                "delta_market_value": round(-current_mv, 2),
                "error": str(exc),
            })

    queued = []
    for sym, w in target_weights.items():
        target_mv = equity * float(w)
        current_mv = _position_market_value(current[sym]) if sym in current else 0.0
        delta = target_mv - current_mv
        if abs(delta) < rebalance_floor:
            actions.append({
                "symbol": sym,
                "action": "hold",
                "current_market_value": round(current_mv, 2),
                "target_market_value": round(target_mv, 2),
                "delta_market_value": round(delta, 2),
            })
            continue
        side = "buy" if delta > 0 else "sell"
        queued.append({
            "symbol": sym,
            "side": side,
            "notional": round(float(abs(delta)), 2),
            "current_market_value": round(current_mv, 2),
            "target_market_value": round(target_mv, 2),
            "delta_market_value": round(delta, 2),
        })

    # Execute sells first, then buys (frees up cash for new positions)
    for item in [x for x in queued if x["side"] == "sell"] + [x for x in queued if x["side"] == "buy"]:
        try:
            order = client.submit_order(item["symbol"], item["notional"], item["side"])
            if order is not None:
                orders.append(order)
                # Track new/updated positions in state
                if state and item["side"] == "buy":
                    price_est = item["target_market_value"] / max(equity, 1) * 100  # rough
                    pos = current.get(item["symbol"])
                    if pos:
                        state.update_peak_price(item["symbol"], float(pos.get("current_price", 0)))
            actions.append({
                "symbol": item["symbol"],
                "action": item["side"],
                "current_market_value": item["current_market_value"],
                "target_market_value": item["target_market_value"],
                "delta_market_value": item["delta_market_value"],
            })
        except Exception as exc:
            failures.append({
                "symbol": item["symbol"],
                "action": item["side"],
                "notional": item["notional"],
                "current_market_value": item["current_market_value"],
                "target_market_value": item["target_market_value"],
                "delta_market_value": item["delta_market_value"],
                "error": str(exc),
            })

    action_summary = {"hold": 0, "buy": 0, "sell": 0, "close": 0}
    for item in actions:
        action_summary[item["action"]] = action_summary.get(item["action"], 0) + 1

    return {
        "equity": equity,
        "positions_seen": len(positions_raw),
        "action_summary": action_summary,
        "orders_submitted": len(orders),
        "orders_failed": len(failures),
        "actions": actions,
        "orders": orders,
        "failures": failures,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SCHEDULING
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_hhmm(value: str) -> dt_time:
    """Parse HH:MM strings used by the realtime schedule."""
    hour_text, minute_text = value.strip().split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid time: {value!r}")
    return dt_time(hour=hour, minute=minute)


def _calendar_run_slots(client: AlpacaClient, market_tz: ZoneInfo, start_day, end_day):
    """Build run slots from Alpaca's trading calendar, including early closes."""
    slots = []
    for session in client.get_calendar(start_day.isoformat(), end_day.isoformat()):
        trade_day = datetime.strptime(session["date"], "%Y-%m-%d").date()
        open_time = _parse_hhmm(session["open"])
        close_time = _parse_hhmm(session["close"])
        slots.append(datetime.combine(trade_day, open_time, tzinfo=market_tz) + timedelta(minutes=15))
        # Midday check
        mid_hour = (open_time.hour + close_time.hour) // 2
        slots.append(datetime.combine(trade_day, dt_time(hour=mid_hour), tzinfo=market_tz))
        # Pre-close
        slots.append(datetime.combine(trade_day, close_time, tzinfo=market_tz) - timedelta(minutes=30))
    return sorted(slots)


def _weekday_run_slots(run_times, market_tz: ZoneInfo, start_day, end_day):
    """Fallback schedule when no exchange calendar is available."""
    slots = []
    day = start_day
    while day <= end_day:
        if day.weekday() < 5:
            for run_time in run_times:
                slots.append(datetime.combine(day, run_time, tzinfo=market_tz))
        day += timedelta(days=1)
    return sorted(slots)


def next_scheduled_run(now_utc: datetime, market_timezone: str, run_times, client: AlpacaClient = None):
    """Return the next scheduled runtime in the market timezone."""
    market_tz = ZoneInfo(market_timezone)
    now_local = now_utc.astimezone(market_tz)
    start_day = now_local.date()
    end_day = start_day + timedelta(days=10)

    slots = []
    if client is not None:
        try:
            slots = _calendar_run_slots(client, market_tz, start_day, end_day)
        except Exception:
            slots = []
    if not slots:
        slots = _weekday_run_slots(run_times, market_tz, start_day, end_day)

    for slot in slots:
        if slot >= now_local:
            return slot
    return _weekday_run_slots(run_times, market_tz, end_day + timedelta(days=1), end_day + timedelta(days=10))[0]


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE CYCLE (main trading logic)
# ═══════════════════════════════════════════════════════════════════════════════

def run_live_cycle(
    bundle: Dict[str, Any],
    client: AlpacaClient,
    state: TradingState,
    period: str,
    interval: str,
    start_date: str = None,
    end_date: str = None,
):
    """Run one complete trading cycle: risk check -> signals -> rebalance."""
    state.cycle_count += 1
    print_section(f"LIVE CYCLE #{state.cycle_count}")
    print_metric("Time (UTC)", datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S"))

    # ── STEP 1: Risk check on existing positions ─────────────────────────────
    if client is not None:
        account = client.get_account()
        equity = float(account.get("equity", 0.0))
        state.update_equity_peak(equity)

        # Drawdown circuit breaker
        if state.is_drawdown_halted(equity):
            dd_pct = (state.equity_peak - equity) / state.equity_peak if state.equity_peak > 0 else 0
            print_metric("STATUS", colour(
                f"DRAWDOWN HALT (portfolio {dd_pct:.1%} below peak ${state.equity_peak:,.0f})", "r"))
            print_metric("Cooldown until", state.last_drawdown_halt or "N/A")
            state.save()
            return

        print_metric("Broker equity", f"${equity:,.2f}")
        print_metric("Equity peak", f"${state.equity_peak:,.2f}")
        print_metric("Realized P&L", f"${state.total_realized_pnl:,.2f}")

    # ── STEP 2: Generate signals ─────────────────────────────────────────────
    out = build_live_signals(
        bundle,
        period=period,
        interval=interval,
        state=state,
        start_date=start_date,
        end_date=end_date,
    )
    print_metric("Timestamp", out["timestamp"])
    print_metric("Regime", colour(out["regime"]["regime"].upper(), "c"))
    print_metric("Market vol (20d ann.)", f"{out['regime'].get('recent_vol', 0):.1%}")
    print_metric("Vol scale", f"{out['regime']['vol_scale']:.2f}x")

    # Show top signals (raw vs filtered)
    top_raw = sorted(out["raw_signals"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    top_filtered = sorted(out["signals"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    print_metric("Top raw signals", ", ".join(f"{k}:{v:+.4f}" for k, v in top_raw))
    print_metric("Top filtered signals", ", ".join(f"{k}:{v:+.4f}" for k, v in top_filtered))

    # Show signal actions
    active = sum(1 for m in out["signal_meta"].values() if m.get("action") == "active")
    filtered = sum(1 for m in out["signal_meta"].values() if m.get("action", "").startswith("filtered"))
    reduced = sum(1 for m in out["signal_meta"].values() if m.get("action", "").startswith("reduced"))
    print_metric("Signal actions", f"{active} active, {filtered} filtered, {reduced} reduced")

    # Show target weights
    nonzero_weights = {k: v for k, v in out["target_weights"].items() if v > 0.01}
    top_weights = sorted(nonzero_weights.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print_metric("Top weights", ", ".join(f"{k}:{v:.1%}" for k, v in top_weights))

    # Show entry timing scores for top signals
    entry_scores = out.get("entry_scores", {})
    approved_entries = {t: s for t, s in entry_scores.items() if s.get("approved")}
    blocked_entries = {t: s for t, s in entry_scores.items()
                       if not s.get("approved") and s.get("score", 0) > 0}
    if approved_entries:
        top_approved = sorted(approved_entries.items(), key=lambda kv: kv[1]["score"], reverse=True)[:5]
        print_metric("Entry approved",
                     colour(", ".join(f"{t}:{s['score']:.2f}" for t, s in top_approved), "g"))
    if blocked_entries:
        top_blocked = sorted(blocked_entries.items(), key=lambda kv: kv[1]["score"], reverse=True)[:5]
        print_metric("Entry blocked",
                     colour(", ".join(f"{t}:{s['score']:.2f}({s.get('reason','')})" for t, s in top_blocked), "y"))

    # ── STEP 3: Risk management on existing positions ────────────────────────
    if client is not None:
        risk = check_position_risk(client, state, out["latest_prices"])
        if risk["exits"]:
            print_metric("Risk exits triggered", colour(str(len(risk["exits"])), "r"))
            for ex in risk["exits"]:
                print_metric(
                    f"  EXIT {ex['symbol']}",
                    f"{ex['reason']} (P&L: {ex['unrealized_pct']:+.2%})",
                )
            exit_results = execute_risk_exits(client, state, risk["exits"])
            for r in exit_results:
                status_colour = "g" if r["status"] == "closed" else "r"
                pnl_str = f" P&L: ${r.get('pnl', 0):+,.2f}" if "pnl" in r else ""
                print_metric(f"  {r['symbol']}", colour(f"{r['status']}{pnl_str}", status_colour))

            for ex in risk["exits"]:
                out["target_weights"].pop(ex["symbol"], None)
        else:
            print_metric("Risk check", colour("All positions OK", "g"))

    # ── STEP 3b: AI position review — model re-evaluates every held position ─
    if client is not None and cfg.RT_AI_EXIT_ENABLED:
        positions_raw = client.get_positions()
        held_symbols = [p["symbol"] for p in positions_raw]
        if held_symbols:
            reviews = ai_review_positions(
                bundle, out.get("live_data", {}), state, held_symbols,
            )
            ai_exits = []
            ai_reduces = []
            ai_holds = []
            for sym, review in reviews.items():
                if review["action"] == "ai_exit":
                    ai_exits.append((sym, review))
                elif review["action"] == "ai_reduce":
                    ai_reduces.append((sym, review))
                else:
                    ai_holds.append((sym, review))

            if ai_exits:
                print_metric("AI exits", colour(str(len(ai_exits)), "r"))
                for sym, review in ai_exits:
                    print_metric(
                        f"  AI EXIT {sym}",
                        colour(f"{review['reason']} (pred={review['prediction']:+.4f}, "
                               f"P&L={review['unrealized_pct']:+.2%})", "r"),
                    )
                    # Close the position
                    try:
                        client.close_position(sym)
                        pnl = state.close_position(sym, out["latest_prices"].get(sym, 0))
                        print_metric(f"    {sym}", colour(f"CLOSED (realized P&L: ${pnl:+,.2f})", "g"))
                    except Exception as exc:
                        print_metric(f"    {sym}", colour(f"CLOSE FAILED: {exc}", "r"))
                    out["target_weights"].pop(sym, None)

            if ai_reduces:
                print_metric("AI reduces", colour(str(len(ai_reduces)), "y"))
                for sym, review in ai_reduces:
                    print_metric(
                        f"  REDUCE {sym}",
                        f"{review['reason']} (pred={review['prediction']:+.4f})",
                    )
                    # Halve the target weight for this position
                    if sym in out["target_weights"]:
                        out["target_weights"][sym] *= 0.5

            if ai_holds:
                # Rank held positions by hold quality for reallocation
                if cfg.RT_POSITION_RANK_REALLOC and len(ai_holds) > 1:
                    ranked = sorted(
                        ai_holds,
                        key=lambda x: float(x[1].get("hold_score", x[1].get("prediction", 0.0))),
                        reverse=True,
                    )
                    best = ranked[0]
                    worst = ranked[-1]
                    best_score = float(best[1].get("hold_score", best[1].get("prediction", 0.0)))
                    worst_score = float(worst[1].get("hold_score", worst[1].get("prediction", 0.0)))
                    print_metric("Best position",
                                 colour(f"{best[0]} (score={best_score:+.4f})", "g"))
                    print_metric("Weakest position",
                                 f"{worst[0]} (score={worst_score:+.4f})")

                    # Reallocate: boost top positions, trim bottom ones
                    for i, (sym, review) in enumerate(ranked):
                        if sym not in out["target_weights"]:
                            continue
                        rank_frac = i / max(len(ranked) - 1, 1)  # 0=best, 1=worst
                        # Best gets 1.2x, worst gets 0.8x
                        realloc_mult = 1.2 - 0.4 * rank_frac
                        out["target_weights"][sym] *= realloc_mult

            print_metric("AI holds", f"{len(ai_holds)} positions maintained")
        else:
            print_metric("AI review", "No positions to review")

    # ── STEP 4: Rebalance ────────────────────────────────────────────────────
    if client is not None:
        reb = rebalance_broker(client, out["target_weights"], state)
        print_metric("Open positions", reb["positions_seen"])
        print_metric(
            "Action summary",
            ", ".join(f"{k}:{v}" for k, v in reb["action_summary"].items() if v > 0) or "none",
        )
        print_metric("Orders submitted", reb["orders_submitted"])
        if reb["orders_failed"] > 0:
            first = reb["failures"][0]
            print_metric(
                "Order failures",
                colour(f"{reb['orders_failed']} (first: {first['symbol']} {first['action']} -> {first['error']})", "r"),
            )

        # Sync position tracking state with broker
        positions_raw = client.get_positions()
        for pos in positions_raw:
            sym = pos["symbol"]
            if sym not in state.positions:
                state.record_position(
                    sym,
                    float(pos.get("avg_entry_price", 0)),
                    float(pos.get("qty", 0)),
                )
            state.update_peak_price(sym, float(pos.get("current_price", 0)))
    else:
        print_metric("Execution", "Disabled (mode=signals)")

    state.save()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP + API
# ═══════════════════════════════════════════════════════════════════════════════

def run_loop(
    bundle_path: Path,
    mode: str,
    period: str,
    interval: str,
    market_timezone: str,
    run_times,
    start_date: str = None,
    end_date: str = None,
):
    bundle = load_model_bundle(bundle_path)
    state = TradingState()

    print_header("REALTIME AI TRADER (v2)")
    print_metric("Model bundle", bundle_path)
    print_metric("Universe", ", ".join(bundle["settings"]["universe"]))
    print_metric("Mode", mode)
    print_metric("Interval", interval)
    if start_date or end_date:
        print_metric("Date window", f"{start_date or 'None'} -> {end_date or 'None'}")
    else:
        print_metric("Period", period)
    print_metric("Schedule TZ", market_timezone)
    print_metric("Run times", ", ".join(t.strftime("%H:%M") for t in run_times))
    print_metric("Stop loss", f"{cfg.RT_STOP_LOSS_PCT:.0%}")
    print_metric("Trailing stop", f"{cfg.RT_TRAILING_STOP_PCT:.0%} (activate at +{cfg.RT_TRAILING_STOP_ACTIVATE:.0%})")
    print_metric("Take profit", f"{cfg.RT_TAKE_PROFIT_PCT:.0%}")
    print_metric("Drawdown halt", f"{cfg.RT_MAX_DRAWDOWN_HALT:.0%}")
    print_metric("Signal smoothing", f"{cfg.RT_SIGNAL_SMOOTHING}-cycle EMA")
    print_metric("Min signal strength", f"{cfg.RT_MIN_SIGNAL_STRENGTH}")
    print_metric("Ensemble agreement", f"{cfg.RT_ENSEMBLE_AGREEMENT_MIN:.0%}")
    print_metric("Vol target", f"{cfg.RT_VOL_TARGET:.0%}")
    print_metric("Cash buffer", f"{cfg.RT_CASH_BUFFER_PCT:.0%}")
    if state.cycle_count > 0:
        print_metric("Resuming from", f"cycle #{state.cycle_count}")
        print_metric("Tracked positions", len(state.positions))

    client = None
    if mode in {"paper", "live"}:
        client = AlpacaClient(mode)

    print_metric("Startup run", "Executing immediate cycle")
    run_live_cycle(
        bundle, client, state, period, interval,
        start_date=start_date, end_date=end_date,
    )

    while True:
        next_run = next_scheduled_run(
            datetime.now(ZoneInfo("UTC")),
            market_timezone=market_timezone,
            run_times=run_times,
            client=client,
        )
        sleep_seconds = max(0.0, (next_run - datetime.now(next_run.tzinfo)).total_seconds())
        print_metric("Next run", next_run.strftime("%Y-%m-%d %H:%M:%S %Z"))
        if sleep_seconds > 0:
            print_metric("Sleep sec", f"{int(round(sleep_seconds)):,}")
            time.sleep(sleep_seconds)

        run_live_cycle(
            bundle, client, state, period, interval,
            start_date=start_date, end_date=end_date,
        )


def run_api_server(
    bundle_path: Path,
    host: str,
    port: int,
    mode: str,
    period: str,
    interval: str,
    start_date: str = None,
    end_date: str = None,
):
    bundle = load_model_bundle(bundle_path)
    state = TradingState()
    client = AlpacaClient(mode) if mode in {"paper", "live"} else None

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: Dict[str, Any]):
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"ok": True, "mode": mode, "cycle": state.cycle_count})
                return
            if self.path == "/signal":
                try:
                    out = build_live_signals(
                        bundle, period=period, interval=interval,
                        state=state, start_date=start_date, end_date=end_date,
                    )
                    self._send_json(200, out)
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                return
            if self.path == "/state":
                self._send_json(200, {
                    "positions": state.positions,
                    "equity_peak": state.equity_peak,
                    "cycle_count": state.cycle_count,
                    "total_realized_pnl": state.total_realized_pnl,
                    "last_drawdown_halt": state.last_drawdown_halt,
                })
                return
            self._send_json(404, {"ok": False, "error": "Not Found"})

        def do_POST(self):
            if self.path == "/rebalance":
                if client is None:
                    self._send_json(400, {"ok": False, "error": "Broker mode disabled."})
                    return
                try:
                    out = build_live_signals(
                        bundle, period=period, interval=interval,
                        state=state, start_date=start_date, end_date=end_date,
                    )
                    # Risk check first
                    risk = check_position_risk(client, state, out["latest_prices"])
                    exit_results = []
                    if risk["exits"]:
                        exit_results = execute_risk_exits(client, state, risk["exits"])
                    reb = rebalance_broker(client, out["target_weights"], state)
                    state.save()
                    self._send_json(200, {
                        "ok": True,
                        "signal": out,
                        "risk_exits": exit_results,
                        "rebalance": reb,
                    })
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                return
            if self.path == "/reset-halt":
                state.last_drawdown_halt = None
                state.save()
                self._send_json(200, {"ok": True, "message": "Drawdown halt cleared"})
                return
            self._send_json(404, {"ok": False, "error": "Not Found"})

    print_header("REALTIME API SERVER (v2)")
    print_metric("Endpoint", f"http://{host}:{port}")
    print_metric("Mode", mode)
    print_metric("Routes", "GET /health /signal /state, POST /rebalance /reset-halt")
    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Run realtime trading using saved model bundle.")
    p.add_argument("--bundle", type=Path, default=cfg.MODEL_BUNDLE_PATH, help="Path to model bundle .pkl")
    p.add_argument("--mode", choices=["signals", "paper", "live"], default="signals")
    p.add_argument("--api", action="store_true", help="Run as HTTP API server instead of loop mode")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--period", default=cfg.REALTIME_BAR_PERIOD)
    p.add_argument("--interval", default=cfg.REALTIME_BAR_INTERVAL)
    p.add_argument("--market-timezone", default=cfg.REALTIME_MARKET_TIMEZONE)
    p.add_argument(
        "--run-times",
        nargs="+",
        default=list(cfg.REALTIME_RUN_TIMES),
        help="Realtime loop run times in market timezone, e.g. 09:45 12:00 15:30",
    )
    p.add_argument("--start-date", default=None, help="Optional explicit start date; leave unset for rolling period")
    p.add_argument("--end-date", default=None, help="Optional explicit end date; leave unset for rolling period")
    args = p.parse_args()
    args.run_times = tuple(_parse_hhmm(value) for value in args.run_times)
    return args


def main():
    load_env_file(Path(__file__).parent / ".env")
    args = parse_args()
    if not args.bundle.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {args.bundle}. Run `python main.py` first to train and save it."
        )
    if args.api:
        run_api_server(
            args.bundle,
            args.host,
            args.port,
            args.mode,
            args.period,
            args.interval,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    else:
        run_loop(
            args.bundle,
            args.mode,
            args.period,
            args.interval,
            args.market_timezone,
            args.run_times,
            start_date=args.start_date,
            end_date=args.end_date,
        )


if __name__ == "__main__":
    main()
