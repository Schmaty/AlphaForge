"""
AI Stock Trader — Configuration
================================
All hyperparameters, feature settings, and output paths.
Edit this file to tune the model without touching any other code.

No external dependencies beyond numpy/pandas/matplotlib.
"""

from pathlib import Path

# Anchor all paths to this file's directory so the project runs correctly
# from any working directory (cron, CI, Docker, etc.)
_BASE = Path(__file__).parent

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = _BASE / "outputs"
DATA_DIR    = _BASE / "data"
REPORT_PATH = OUTPUT_DIR / "trading_report.txt"
CHART_PATH  = OUTPUT_DIR / "trading_charts.png"
MODEL_BUNDLE_PATH = OUTPUT_DIR / "model_bundle.pkl"

# ── Universe ───────────────────────────────────────────────────────────────────
UNIVERSE = [
    "NVDA", "MSFT", "AAPL", "META",
    "GOOGL", "AMZN", "AVGO", "TSM",
    "LLY", "UNH", "V", "MA",
    "COST", "NFLX", "CRM", "AMD",
    "ISRG", "NOW", "UBER", "GE",
]

BENCHMARK = "SPY"

# Simulation dates — 2004-01-01 start gives feature indicators enough
# warmup history before the 2005+ effective trading period.
# All universe tickers were publicly listed before 2005.
START_DATE = "1999-10-01"
END_DATE   = "2025-12-31"

# Train / test split (most-recent N% used for out-of-sample test)
TEST_SPLIT = 0.50

# ── Feature Engineering ────────────────────────────────────────────────────────
LOOKBACK_WINDOW = 40   # trading days of history fed to the model as features
FORWARD_DAYS    = 3    # forward return horizon for target (prediction horizon)

# ── Neural Network (pure-NumPy MLP) ───────────────────────────────────────────
HIDDEN_LAYERS = (256, 128, 64)  # neurons per hidden layer (tuple — immutable)
DROPOUT       = 0.20
ACTIVATION    = "leaky_relu"    # "relu" | "leaky_relu" | "tanh" | "elu"

# Training
EPOCHS              = 60
BATCH_SIZE          = 128
LEARNING_RATE       = 6e-4
WEIGHT_DECAY        = 1e-5
LR_DECAY            = 0.993     # per-epoch multiplicative LR decay
EARLY_STOP_PATIENCE = 15
GRAD_CLIP           = 2.0

# ── Ensemble ───────────────────────────────────────────────────────────────────
ENSEMBLE_MODELS = 5     # train N models, average predictions
BOOTSTRAP_RATIO = 0.88  # fraction of train data each model sees

# ── Signal & Position Sizing ──────────────────────────────────────────────────
SIGNAL_THRESHOLD = 0.005    # minimum predicted score to trigger trade
POSITION_SIZING  = "risk_parity"  # "equal" | "risk_parity" | "momentum"
MAX_POSITION_PCT = 0.20     # max weight per ticker (20%)
REBALANCE_DAYS   = 3        # trading days between rebalances
SIGNAL_BLEND     = 0.75     # model signal weight (75% model / 25% equal)
SIGNAL_SCALE     = 8.0      # sharper softmax differentiation

# ── Risk Management ───────────────────────────────────────────────────────────
STOP_LOSS_PCT          = 0.06
TAKE_PROFIT_PCT        = 0.18
MAX_PORTFOLIO_DRAWDOWN = 0.12
TRANSACTION_COST_BPS   = 10   # basis points per trade leg

# ── Financial Parameters ──────────────────────────────────────────────────────
RISK_FREE_RATE = 0.04  # annual risk-free rate for Sharpe / Sortino / Treynor

# ── Reporting ──────────────────────────────────────────────────────────────────
CHART_DPI   = 150
CHART_STYLE = "dark_background"

# ── Realtime Trading Defaults ───────────────────────────────────────────────────
REALTIME_BAR_INTERVAL = "1d"   # match backtest/training daily timeframe
REALTIME_BAR_PERIOD   = "1y"  # safer rolling window for feature warmup + lookback
REALTIME_MARKET_TIMEZONE = "America/New_York"
REALTIME_RUN_TIMES = ("09:45", "12:00", "15:30")  # morning, midday, pre-close

# ── Realtime Risk Management ─────────────────────────────────────────────────
RT_STOP_LOSS_PCT           = 0.05   # close position if it drops 5% from entry
RT_TRAILING_STOP_PCT       = 0.03   # trailing stop: lock in gains at 3% from peak
RT_TRAILING_STOP_ACTIVATE  = 0.02   # only activate trailing stop after 2% gain
RT_TAKE_PROFIT_PCT         = 0.15   # take profit at 15% gain
RT_MAX_DRAWDOWN_HALT       = 0.08   # halt all trading if portfolio drops 8% from peak
RT_DRAWDOWN_COOLDOWN_HOURS = 24     # hours to wait after drawdown halt before resuming
RT_MIN_SIGNAL_STRENGTH     = 0.008  # ignore signals weaker than this
RT_STRONG_SIGNAL_THRESHOLD = 0.025  # scale up position for signals above this

# ── Realtime Signal Enhancement ──────────────────────────────────────────────
RT_SIGNAL_SMOOTHING        = 3      # EMA smoothing of signals across cycles
RT_ENSEMBLE_AGREEMENT_MIN  = 0.6    # fraction of models that must agree on direction
RT_VOL_SCALE_ENABLED       = True   # scale position size inversely with volatility
RT_VOL_LOOKBACK_DAYS       = 20     # days for realtime volatility calculation
RT_VOL_TARGET              = 0.15   # target annualized vol for position sizing
RT_CASH_BUFFER_PCT         = 0.05   # keep 5% in cash for opportunities / margin

# ── Realtime Entry Timing ─────────────────────────────────────────────────────
RT_ENTRY_SCORE_MIN         = 0.4    # minimum entry quality score (0-1) to actually buy
RT_RSI_OVERBOUGHT          = 72     # don't enter longs above this RSI
RT_RSI_OVERSOLD            = 30     # ideal long entry zone below this RSI
RT_BB_UPPER_AVOID          = 0.90   # don't enter longs when price is above 90% of BB range
RT_MACD_CONFIRM_REQUIRED   = True   # require MACD histogram to confirm signal direction
RT_VOLUME_CONFIRM_MIN      = 0.8    # min volume ratio (vs 20d avg) to confirm entry

# ── AI Position Review ───────────────────────────────────────────────────────
RT_AI_EXIT_ENABLED         = True   # use model to decide whether to hold each position
RT_AI_EXIT_SIGNAL_FLIP     = -0.005 # exit if model prediction flips this far negative
RT_AI_EXIT_DECAY_CYCLES    = 4      # exit if signal decays for this many consecutive cycles
RT_POSITION_RANK_REALLOC   = True   # reallocate from weak positions to strong ones

# ── Realtime State Persistence ───────────────────────────────────────────────
RT_STATE_PATH = OUTPUT_DIR / "realtime_state.json"
