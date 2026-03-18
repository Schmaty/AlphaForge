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
LOOKBACK_WINDOW = 40           # trading days fed to the model

# ── Neural Network (pure-NumPy MLP) ───────────────────────────────────────────
HIDDEN_LAYERS = [256, 128, 64] # neurons per hidden layer
DROPOUT       = 0.25
ACTIVATION    = "leaky_relu"   # "relu" | "leaky_relu" | "tanh" | "elu"

# Training
EPOCHS             = 50
BATCH_SIZE         = 128
LEARNING_RATE      = 5e-4
WEIGHT_DECAY       = 1e-5
LR_DECAY           = 0.995     # per-epoch multiplicative decay
EARLY_STOP_PATIENCE = 12
GRAD_CLIP          = 2.0

# ── Ensemble ───────────────────────────────────────────────────────────────────
ENSEMBLE_MODELS    = 3         # train N models, average predictions
BOOTSTRAP_RATIO    = 0.85      # fraction of train data each model sees

# ── Signal & Position Sizing ──────────────────────────────────────────────────
SIGNAL_THRESHOLD   = 0.005     # minimum predicted score to trigger trade
POSITION_SIZING    = "risk_parity"   # "equal" | "risk_parity" | "momentum"
MAX_POSITION_PCT   = 0.15      # max weight per ticker

# ── Risk Management ───────────────────────────────────────────────────────────
STOP_LOSS_PCT          = 0.06
TAKE_PROFIT_PCT        = 0.18
MAX_PORTFOLIO_DRAWDOWN = 0.12
TRANSACTION_COST_BPS   = 10    # basis points per trade

# ── Reporting ──────────────────────────────────────────────────────────────────
CHART_DPI   = 150
CHART_STYLE = "dark_background"
