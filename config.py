"""
AI Stock Trader — Configuration
================================
All hyperparameters, feature settings, real stock calibration data, and output paths.
Edit this file to tune the model without touching any other code.

No external dependencies beyond numpy/pandas/matplotlib.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
REPORT_PATH = OUTPUT_DIR / "trading_report.txt"
CHART_PATH = OUTPUT_DIR / "trading_charts.png"

# ── Universe ───────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "JNJ",
    "UNH", "XOM", "PG", "HD", "MA",
    "LLY", "ABBV", "MRK", "AVGO", "PEP",
]

BENCHMARK = "SPY"

# Simulation dates
START_DATE = "2018-01-01"
END_DATE   = "2025-12-31"

# Train / test split (most-recent N% used for out-of-sample test)
TEST_SPLIT = 0.30

# ── Real Stock Calibration Data (2018-2025 historical estimates) ───────────────
# Each entry: (annual_return%, annual_vol%, beta_to_SPY, sector_id, approx_price_2018)
# Sectors: 0=Tech, 1=Finance, 2=Health, 3=Energy, 4=Consumer
STOCK_PARAMS = {
    "AAPL":  (0.32, 0.30, 1.20, 0, 170),
    "MSFT":  (0.30, 0.28, 1.10, 0, 105),
    "GOOGL": (0.18, 0.30, 1.10, 0, 1050),
    "AMZN":  (0.15, 0.33, 1.20, 0, 1170),
    "NVDA":  (0.65, 0.52, 1.70, 0, 33),
    "META":  (0.20, 0.40, 1.30, 0, 177),
    "TSLA":  (0.55, 0.62, 1.90, 0, 21),
    "JPM":   (0.14, 0.28, 1.10, 1, 106),
    "V":     (0.16, 0.24, 0.95, 1, 121),
    "JNJ":   (0.06, 0.18, 0.65, 2, 140),
    "UNH":   (0.20, 0.24, 0.80, 2, 223),
    "XOM":   (0.08, 0.30, 0.90, 3, 79),
    "PG":    (0.10, 0.18, 0.55, 4, 92),
    "HD":    (0.15, 0.26, 1.00, 4, 189),
    "MA":    (0.18, 0.26, 1.00, 1, 203),
    "LLY":   (0.40, 0.28, 0.60, 2, 85),
    "ABBV":  (0.16, 0.26, 0.70, 2, 97),
    "MRK":   (0.14, 0.22, 0.65, 2, 56),
    "AVGO":  (0.35, 0.35, 1.30, 0, 260),
    "PEP":   (0.08, 0.18, 0.60, 4, 119),
}

# SPY benchmark calibration (2018-2025)
SPY_ANNUAL_RETURN = 0.12   # ~12% CAGR for SPY 2018-2025
SPY_ANNUAL_VOL    = 0.18   # ~18% annual vol

# Sector correlations (within-sector stocks are more correlated)
SECTOR_NAMES = {0: "Technology", 1: "Financials", 2: "Healthcare", 3: "Energy", 4: "Consumer"}

# ── Feature Engineering ────────────────────────────────────────────────────────
LOOKBACK_WINDOW = 20           # trading days fed to the model

# ── Neural Network (pure-NumPy MLP) ───────────────────────────────────────────
HIDDEN_LAYERS = [512, 256, 128, 64] # original fast architecture
DROPOUT       = 0.30
ACTIVATION    = "leaky_relu"   # "relu" | "leaky_relu" | "tanh" | "elu"

# Training — fast (original speed)
EPOCHS             = 50
BATCH_SIZE         = 128
LEARNING_RATE      = 5e-4
WEIGHT_DECAY       = 1e-5
LR_DECAY           = 0.995     # per-epoch multiplicative decay
EARLY_STOP_PATIENCE = 12
GRAD_CLIP          = 2.0

# ── Ensemble ───────────────────────────────────────────────────────────────────
ENSEMBLE_MODELS    = 3         # fast: 3 models
BOOTSTRAP_RATIO    = 0.85      # fraction of train data each model sees

# ── Signal & Position Sizing ──────────────────────────────────────────────────
SIGNAL_THRESHOLD   = 0.005     # minimum predicted score to trigger trade
POSITION_SIZING    = "risk_parity"   # "equal" | "risk_parity" | "momentum"
MAX_POSITION_PCT   = 0.25      # allow high concentration in winners
SOFTMAX_SCALE      = 10.0      # strong conviction weighting
SIGNAL_BLEND       = 0.80      # blend: 80% model weights, 20% equal weight
VAL_SPLIT_RATIO    = 0.82      # train vs. validation split within training period

# ── Momentum Overlay ─────────────────────────────────────────────────────────
MOMENTUM_WEIGHT    = 0.80      # primary driver: momentum factor
MODEL_WEIGHT       = 0.20      # secondary: ML model (calibrated)
MOMENTUM_LOOKBACK  = 120       # 6-month trailing momentum (strongest factor)

# ── Long/Short Mode ──────────────────────────────────────────────────────────
ALLOW_SHORT        = True      # long/short with momentum signals
SHORT_SCALE        = 0.15      # hedging short book
MAX_SHORT_PCT      = 0.05      # max short weight per ticker
N_SHORT            = 4         # bottom 4 stocks to short
GROSS_LEVERAGE     = 1.12      # conservative leverage
NET_EXPOSURE_RANGE = (0.90, 1.10)  # stay net long

# ── Risk Management ───────────────────────────────────────────────────────────
STOP_LOSS_PCT          = 0.18  # wide stops — avoid whipsaw
TAKE_PROFIT_PCT        = 0.45  # let winners run
MAX_PORTFOLIO_DRAWDOWN = 0.20  # tighter circuit breaker
TRANSACTION_COST_BPS   = 10    # basis points per trade

# ── Rebalancing ──────────────────────────────────────────────────────────────
REBALANCE_DAYS     = 20        # monthly rebalancing (reduce turnover)

# ── Reporting ──────────────────────────────────────────────────────────────────
CHART_DPI   = 150
CHART_STYLE = "dark_background"
