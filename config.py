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
    "AAPL", "MSFT", "AMZN", "NVDA", 
    "IBM", "INTC", "JPM", "BAC", 
    "JNJ", "UNH", "XOM", "PG", 
    "HD", "GS", "LLY", "KO", 
    "MRK", "WMT", "PEP", "AXP" 
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
REALTIME_POLL_SECONDS = 10800 # 3 hours
