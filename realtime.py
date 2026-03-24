#!/usr/bin/env python3
"""
Realtime trader that uses trained models from main.py.

Features:
  - Loads persisted ensemble bundle (outputs/model_bundle.pkl)
  - Pulls live/delayed bars from Yahoo Finance
  - Reuses backtest weight logic for portfolio targets
  - Optional Alpaca execution (paper or live)
  - Small local HTTP API for online usage
"""

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Any
from urllib.error import HTTPError, URLError
from urllib import request as urlrequest

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
)
from utils import _signal_to_weights


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
        # Keep existing exported env vars as highest priority.
        os.environ.setdefault(key, value)


def fetch_live_ohlcv(tickers, period=None, interval="1d", start_date=None, end_date=None):
    """
    Fetch OHLCV bars for all tickers via yfinance.

    If start/end are provided, they are used (matching backtest date-window style).
    Otherwise period+interval mode is used.
    """
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
        # yfinance can return multi-index columns depending on version/options.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in df.columns:
                raise RuntimeError(f"Missing column {c} for {t}")
        data[t] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    return data


def build_live_signals(
    bundle: Dict[str, Any],
    period: str,
    interval: str,
    start_date: str = None,
    end_date: str = None,
):
    """Generate one latest signal per ticker from live bars."""
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

    preds = {}
    latest_ts = None
    for t in tickers:
        df = live[t]
        feat_raw = add_features(df["Close"], df["High"], df["Low"], df["Volume"])
        if t not in norm_stats:
            raise RuntimeError(f"Ticker {t} missing normalisation stats in bundle.")
        mu, sigma = norm_stats[t]
        feat_norm = apply_normalisation(feat_raw, mu, sigma)
        if len(feat_norm) < lookback:
            raise RuntimeError(f"Not enough bars for {t}: have {len(feat_norm)} need {lookback}")

        x = feat_norm.iloc[-lookback:].values.flatten().astype(np.float64).reshape(1, -1)
        pred = 0.0
        for m, w in zip(models, ensemble_weights):
            pred += float(w) * float(m.predict(x)[0])
        preds[t] = pred
        latest_ts = feat_norm.index[-1]

    signal_series = pd.Series(preds).sort_index()
    weights = _signal_to_weights(signal_series, list(signal_series.index), cfg)
    return {
        "timestamp": str(latest_ts),
        "signals": {k: float(v) for k, v in signal_series.to_dict().items()},
        "target_weights": {k: float(v) for k, v in weights.items()},
    }


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


def _position_market_value(position: Dict[str, Any]) -> float:
    """Return signed market value so short positions are treated as negative."""
    market_value = float(position.get("market_value", 0.0))
    side = str(position.get("side", "")).lower()
    if side == "short":
        return -abs(market_value)
    return abs(market_value)


def rebalance_broker(client: AlpacaClient, target_weights: Dict[str, float]):
    """Rebalance against live Alpaca equity + positions."""
    account = client.get_account()
    equity = float(account.get("equity", 0.0))
    positions_raw = client.get_positions()
    current = {p["symbol"]: p for p in positions_raw}
    rebalance_floor = max(50.0, 0.001 * max(equity, 1.0))
    managed_symbols = {sym for sym, weight in target_weights.items() if float(weight) > 0.0}

    orders = []
    failures = []
    actions = []

    # Close positions the model is not managing before adding new exposure.
    for sym, position in current.items():
        if sym in managed_symbols:
            continue
        current_mv = _position_market_value(position)
        if abs(current_mv) < 1.0:
            continue
        try:
            order = client.close_position(sym)
            orders.append(order)
            actions.append(
                {
                    "symbol": sym,
                    "action": "close",
                    "current_market_value": round(current_mv, 2),
                    "target_market_value": 0.0,
                    "delta_market_value": round(-current_mv, 2),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "symbol": sym,
                    "action": "close",
                    "current_market_value": round(current_mv, 2),
                    "target_market_value": 0.0,
                    "delta_market_value": round(-current_mv, 2),
                    "error": str(exc),
                }
            )

    queued = []
    for sym, w in target_weights.items():
        target_mv = equity * float(w)
        current_mv = _position_market_value(current[sym]) if sym in current else 0.0
        delta = target_mv - current_mv
        if abs(delta) < rebalance_floor:
            actions.append(
                {
                    "symbol": sym,
                    "action": "hold",
                    "current_market_value": round(current_mv, 2),
                    "target_market_value": round(target_mv, 2),
                    "delta_market_value": round(delta, 2),
                }
            )
            continue
        side = "buy" if delta > 0 else "sell"
        queued.append(
            {
                "symbol": sym,
                "side": side,
                "notional": round(float(abs(delta)), 2),
                "current_market_value": round(current_mv, 2),
                "target_market_value": round(target_mv, 2),
                "delta_market_value": round(delta, 2),
            }
        )

    for item in [x for x in queued if x["side"] == "sell"] + [x for x in queued if x["side"] == "buy"]:
        try:
            order = client.submit_order(item["symbol"], item["notional"], item["side"])
            if order is not None:
                orders.append(order)
            actions.append(
                {
                    "symbol": item["symbol"],
                    "action": item["side"],
                    "current_market_value": item["current_market_value"],
                    "target_market_value": item["target_market_value"],
                    "delta_market_value": item["delta_market_value"],
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "symbol": item["symbol"],
                    "action": item["side"],
                    "notional": item["notional"],
                    "current_market_value": item["current_market_value"],
                    "target_market_value": item["target_market_value"],
                    "delta_market_value": item["delta_market_value"],
                    "error": str(exc),
                }
            )

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


def run_loop(
    bundle_path: Path,
    mode: str,
    poll_seconds: int,
    period: str,
    interval: str,
    start_date: str = None,
    end_date: str = None,
):
    bundle = load_model_bundle(bundle_path)
    print_header("REALTIME AI TRADER")
    print_metric("Model bundle", bundle_path)
    print_metric("Universe", ", ".join(bundle["settings"]["universe"]))
    print_metric("Mode", mode)
    print_metric("Interval", interval)
    if start_date or end_date:
        print_metric("Date window", f"{start_date or 'None'} -> {end_date or 'None'}")
    else:
        print_metric("Period", period)
    print_metric("Poll sec", poll_seconds)

    client = None
    if mode in {"paper", "live"}:
        client = AlpacaClient(mode)

    while True:
        print_section("LIVE CYCLE")
        out = build_live_signals(
            bundle,
            period=period,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
        )
        print_metric("Timestamp", out["timestamp"])
        top = sorted(out["signals"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        print_metric("Top signals", ", ".join(f"{k}:{v:+.4f}" for k, v in top))

        if client is not None:
            reb = rebalance_broker(client, out["target_weights"])
            print_metric("Broker equity", f"${reb['equity']:,.2f}")
            print_metric("Open positions", reb["positions_seen"])
            print_metric(
                "Action summary",
                ", ".join(f"{k}:{v}" for k, v in reb["action_summary"].items() if v > 0) or "none",
            )
            print_metric("Orders", reb["orders_submitted"])
            if reb["orders_failed"] > 0:
                first = reb["failures"][0]
                print_metric(
                    "Order failures",
                    f"{reb['orders_failed']} (first: {first['symbol']} {first['action']} -> {first['error']})",
                )
        else:
            print_metric("Execution", "Disabled (mode=signals)")

        time.sleep(max(1, poll_seconds))


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
    client = AlpacaClient(mode) if mode in {"paper", "live"} else None

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: Dict[str, Any]):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"ok": True, "mode": mode})
                return
            if self.path == "/signal":
                try:
                    out = build_live_signals(
                        bundle,
                        period=period,
                        interval=interval,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    self._send_json(200, out)
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                return
            self._send_json(404, {"ok": False, "error": "Not Found"})

        def do_POST(self):
            if self.path != "/rebalance":
                self._send_json(404, {"ok": False, "error": "Not Found"})
                return
            if client is None:
                self._send_json(400, {"ok": False, "error": "Broker mode disabled."})
                return
            try:
                out = build_live_signals(
                    bundle,
                    period=period,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                )
                reb = rebalance_broker(client, out["target_weights"])
                self._send_json(200, {"ok": True, "signal": out, "rebalance": reb})
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})

    print_header("REALTIME API SERVER")
    print_metric("Endpoint", f"http://{host}:{port}")
    print_metric("Mode", mode)
    print_metric("Routes", "GET /health, GET /signal, POST /rebalance")
    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


def parse_args():
    p = argparse.ArgumentParser(description="Run realtime trading using saved model bundle.")
    p.add_argument("--bundle", type=Path, default=cfg.MODEL_BUNDLE_PATH, help="Path to model bundle .pkl")
    p.add_argument("--mode", choices=["signals", "paper", "live"], default="signals")
    p.add_argument("--api", action="store_true", help="Run as HTTP API server instead of loop mode")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--poll-seconds", type=int, default=cfg.REALTIME_POLL_SECONDS)
    p.add_argument("--period", default=cfg.REALTIME_BAR_PERIOD)
    p.add_argument("--interval", default=cfg.REALTIME_BAR_INTERVAL)
    p.add_argument("--start-date", default=None, help="Optional explicit start date; leave unset for rolling period")
    p.add_argument("--end-date", default=None, help="Optional explicit end date; leave unset for rolling period")
    return p.parse_args()


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
            args.poll_seconds,
            args.period,
            args.interval,
            start_date=args.start_date,
            end_date=args.end_date,
        )


if __name__ == "__main__":
    main()
