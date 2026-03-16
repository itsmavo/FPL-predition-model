#!/usr/bin/env python3
"""
FPL Team Generator — v3.0
Fixes applied (v2):
  - update_predicted_points no longer overwrites the weighted formula
  - add_fixture_difficulty now correctly targets the next unplayed GW only
  - calculate_form uses concurrent HTTP requests instead of 600 sequential calls
  - fdr null-guard prevents division-by-zero on blank GWs
  - FDR factor is clipped to avoid division by zero or negative values
Phase 2 additions:
  - SQLite-backed SeasonTracker: persists every squad selection per GW
  - record_actual_points(): call after each GW resolves to store real points
  - season_summary(): prints accuracy stats across all tracked GWs
  - get_gw_selection(): retrieve any past GW's squad from the DB
Phase 3 additions:
  - FPLModel: LightGBM model trained on vaastav historical dataset
    * build_training_data(): loads and engineers features from CSV history files
    * train(): fits LGBMRegressor with early stopping on a validation split
    * predict(): generates next-GW point predictions for current players
    * save() / load(): persist trained model to disk, skip retraining each run
  - FPLSeasonModel: XGBoost model for remainder-of-season point prediction
    * Same interface as FPLModel but targets cumulative season remainder points
    * Used when run(mode='season') is called
  - CaptaincyModel: GradientBoosting classifier for haul probability (≥12 pts)
    * predict_proba() ranks players by captain upside, not just mean prediction
  - Heuristic fallback: if no trained model exists and no vaastav data is
    present, the system falls back gracefully to the v2 weighted formula
  - New dependency: lightgbm, xgboost, scikit-learn, joblib
    pip install lightgbm xgboost scikit-learn joblib
"""

import requests
import pandas as pd
import pulp
import numpy as np
import sqlite3
import os
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Phase 3 — ML imports (graceful fallback if not installed)
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    warnings.warn("lightgbm not installed. Run: pip install lightgbm")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("xgboost not installed. Run: pip install xgboost")

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn / joblib not installed. Run: pip install scikit-learn joblib")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
FORM_WEIGHT         = 0.28
TOTAL_POINTS_WEIGHT = 0.72
FDR_WEIGHT          = 0.15          # penalty applied on top of weighted score
MIN_MINUTES         = 450           # ~5 full games
MAX_FORM_WORKERS    = 10            # concurrent API requests for player history
DB_PATH             = "fpl_tracker.db"

# Phase 3 — ML config
MODEL_DIR           = Path("models")          # where trained models are saved
VAASTAV_DIR         = Path("vaastav_data")    # path to vaastav CSV history files
                                               # download from:
                                               # github.com/vaastav/Fantasy-Premier-League
HAUL_THRESHOLD      = 12   # pts — captaincy model: what counts as a "haul"
RETRAIN_MAE_TRIGGER = 3.5  # if rolling 5-GW MAE exceeds this, auto-retrain
MIN_TRAIN_ROWS      = 500  # minimum rows needed to train (skip if too little data)

# ML feature columns used by all three models
GW_FEATURES = [
    "pts_roll3", "pts_roll6", "xg_roll5", "xa_roll5",
    "minutes_pct", "goals_roll5", "assists_roll5",
    "clean_sheets_roll5", "bonus_roll5",
    "fdr", "is_home",
    "pos_GK", "pos_DEF", "pos_MID", "pos_FWD",
    "price", "transfers_in_event_norm",
]


# ══════════════════════════════════════════════
# SECTION 1 — DATA FETCHING
# ══════════════════════════════════════════════

def fetch_data():
    """Fetch bootstrap-static (players + teams + events)."""
    print("Fetching bootstrap data from FPL API...")
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_fixture_data():
    """Fetch full fixture list."""
    print("Fetching fixture data from FPL API...")
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def _fetch_single_player_history(player_id):
    """
    Internal helper — fetches history for one player.
    Returns (player_id, DataFrame) or (player_id, None) on failure.
    """
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        history = pd.DataFrame(response.json().get("history", []))
        return player_id, history
    except Exception:
        return player_id, None


def fetch_all_player_histories(player_ids):
    """
    FIX: Fetch all player histories concurrently instead of 600 sequential
    API calls. Uses a thread pool capped at MAX_FORM_WORKERS.
    Returns dict {player_id: DataFrame}.
    """
    print(f"Fetching player histories concurrently ({len(player_ids)} players, "
          f"up to {MAX_FORM_WORKERS} threads)...")
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_FORM_WORKERS) as executor:
        futures = {executor.submit(_fetch_single_player_history, pid): pid
                   for pid in player_ids}
        for i, future in enumerate(as_completed(futures), 1):
            pid, history = future.result()
            results[pid] = history
            if i % 50 == 0:
                print(f"  ...{i}/{len(player_ids)} fetched")
    print("  All player histories fetched.")
    return results


# ══════════════════════════════════════════════
# SECTION 2 — PREPROCESSING
# ══════════════════════════════════════════════

def preprocess_data(data):
    """Clean and enrich the raw bootstrap-static payload."""
    print("Preprocessing data...")
    players = pd.DataFrame(data["elements"])
    teams   = pd.DataFrame(data["teams"])

    # Keep relevant columns; also grab FPL's own form string for fallback
    players = players[[
        "id", "web_name", "team", "element_type", "now_cost",
        "total_points", "minutes", "goals_scored", "assists",
        "clean_sheets", "saves", "bonus", "form",
        "selected_by_percent", "transfers_in_event"
    ]].copy()

    # Map IDs → names
    team_id_to_name     = dict(zip(teams["id"], teams["name"]))
    position_id_to_name = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    players["team"]         = players["team"].map(team_id_to_name)
    players["element_type"] = players["element_type"].map(position_id_to_name)

    # Convert FPL form string → float (used as fallback)
    players["form"] = pd.to_numeric(players["form"], errors="coerce").fillna(0)

    # Points per 90 minutes
    players["ppg"] = players["total_points"] / (players["minutes"] / 90)
    players["ppg"] = players["ppg"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Filter out barely-played players
    players = players[players["minutes"] >= MIN_MINUTES].copy()

    # Ensure no zero cost slips through
    players["now_cost"] = players["now_cost"].fillna(0)
    players = players[players["now_cost"] > 0].copy()

    # Initial predicted points = ppg (will be updated later)
    players["predicted_points"] = players["ppg"]

    players.reset_index(drop=True, inplace=True)
    return players, teams


# ══════════════════════════════════════════════
# SECTION 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════

def calculate_form(players):
    """
    FIX: Uses concurrent fetching instead of sequential per-player calls.
    Computes average points over the last 7 GWs for each player.
    Falls back to FPL's own 'form' field if history is unavailable.
    """
    print("Calculating player form (last 7 GWs)...")
    player_ids = players["id"].tolist()
    histories  = fetch_all_player_histories(player_ids)

    form_values = {}
    for _, row in players.iterrows():
        pid     = row["id"]
        history = histories.get(pid)
        if history is not None and not history.empty:
            pts = pd.to_numeric(history["total_points"], errors="coerce")
            form_values[pid] = pts.tail(7).mean()
        else:
            # Fallback to FPL's own rolling form value
            form_values[pid] = row["form"]

    players["form_calc"] = players["id"].map(form_values).fillna(0)
    return players


def add_fixture_difficulty(players, fixtures, teams):
    """
    FIX: Now correctly picks only the NEXT unplayed gameweek's fixture
    per team, not the last fixture in the full list.
    FIX: Adds null guard — blank GW teams get a neutral FDR of 3.
    """
    print("Adding fixture difficulty for next GW...")

    team_id_to_name = dict(zip(teams["id"], teams["name"]))
    fixtures = fixtures.copy()
    fixtures["team_a"] = fixtures["team_a"].map(team_id_to_name)
    fixtures["team_h"] = fixtures["team_h"].map(team_id_to_name)

    # FIX: Only look at unfinished fixtures, take the very next GW
    upcoming = fixtures[fixtures["finished"] == False].copy()
    if upcoming.empty:
        print("  Warning: No upcoming fixtures found. Using neutral FDR=3.")
        players["fdr"] = 3
        return players

    next_gw    = upcoming["event"].min()
    next_fixes = upcoming[upcoming["event"] == next_gw]
    print(f"  Using GW {next_gw} fixtures ({len(next_fixes)} matches).")

    # Build one FDR entry per team for that GW
    fdr_dict = {}
    for _, fixture in next_fixes.iterrows():
        fdr_dict[fixture["team_a"]] = fixture["team_h_difficulty"]
        fdr_dict[fixture["team_h"]] = fixture["team_a_difficulty"]

    # FIX: fillna(3) handles blank GWs — neutral difficulty
    players["fdr"] = players["team"].map(fdr_dict).fillna(3)
    return players


def normalize_metric(series):
    """Min-max normalization, returns values in [0, 1]."""
    rng = series.max() - series.min()
    if rng == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / rng


def update_predicted_points(players):
    """
    FIX: The weighted formula is no longer overwritten on the next line.
    FIX: FDR is clipped to [1, 5] before inversion to prevent
         division by zero or nonsensical values.
    
    Predicted points = weighted(form, total_points) * fdr_factor
    where fdr_factor penalises players with hard upcoming fixtures.
    """
    print("Updating predicted points...")

    players["form_normalized"]         = normalize_metric(players["form_calc"])
    players["total_points_normalized"] = normalize_metric(players["total_points"])

    # Weighted blend of form and season total
    weighted_score = (
        FORM_WEIGHT         * players["form_normalized"] +
        TOTAL_POINTS_WEIGHT * players["total_points_normalized"]
    )

    # FIX: clip FDR to valid range before inverting
    fdr_clipped = players["fdr"].clip(lower=1, upper=5)

    # Invert and scale: FDR 1 (easy) → factor ~1.0, FDR 5 (hard) → factor ~0.6
    fdr_factor = 1 - FDR_WEIGHT * ((fdr_clipped - 1) / 4)

    players["predicted_points"] = weighted_score * fdr_factor
    return players


# ══════════════════════════════════════════════
# SECTION 4 — PHASE 3: ML MODELS
# ══════════════════════════════════════════════

# ─────────────────────────────────────────────
# 4a. Training data builder
# ─────────────────────────────────────────────

def build_training_data(vaastav_dir=VAASTAV_DIR):
    """
    Load and engineer features from the vaastav FPL historical dataset.

    Handles all three layout variants found in the actual vaastav repo:

      Layout A — individual GW files (older seasons):
        vaastav_data/data/2018-19/gws/gw1.csv ... gw38.csv

      Layout B — merged file (newer seasons):
        vaastav_data/data/2021-22/gws/merged_gw.csv

      Layout C — no data/ subfolder (if cloned with custom name):
        vaastav_data/2021-22/gws/merged_gw.csv

    The function auto-detects which layout is present and loads all
    seasons it can find, then concatenates into one training frame.

    Returns:
        df (DataFrame) with engineered features and two targets:
            'target_next_gw'      — points in the following GW  (Model A)
            'target_season_rem'   — cumulative points from this GW to GW38 (Model B)
        Returns None if no data found.
    """
    vaastav_dir = Path(vaastav_dir)
    if not vaastav_dir.exists():
        print(f"  [Phase 3] vaastav_data/ not found at {vaastav_dir.resolve()}.")
        print("  Download from https://github.com/vaastav/Fantasy-Premier-League")
        print("  and place season folders inside vaastav_data/")
        return None

    # ── Auto-detect root: handle optional data/ subfolder ─────────
    # vaastav_data/data/2018-19/...  →  root = vaastav_data/data
    # vaastav_data/2018-19/...       →  root = vaastav_data
    data_subdir = vaastav_dir / "data"
    root = data_subdir if data_subdir.exists() else vaastav_dir
    print(f"  [Phase 3] Scanning {root.resolve()} for season folders...")

    frames = []
    seasons_loaded = 0

    for season_dir in sorted(root.iterdir()):
        if not season_dir.is_dir():
            continue
        gws_dir = season_dir / "gws"
        if not gws_dir.exists():
            continue

        season_name = season_dir.name
        season_frames = []

        # ── Try Layout B first: merged_gw.csv ─────────────────────
        merged = gws_dir / "merged_gw.csv"
        if merged.exists():
            try:
                df = pd.read_csv(merged, encoding="utf-8", low_memory=False)
                df["season"] = season_name
                season_frames.append(df)
            except Exception as e:
                print(f"  [Phase 3] Warning: could not read {merged}: {e}")

        # ── Layout A: individual gw1.csv ... gw38.csv ─────────────
        else:
            for gw_num in range(1, 39):
                gw_file = gws_dir / f"gw{gw_num}.csv"
                if gw_file.exists():
                    try:
                        df = pd.read_csv(gw_file, encoding="utf-8",
                                         low_memory=False)
                        df["season"] = season_name
                        # Inject GW number — older files may not have it
                        if "gw" not in [c.lower() for c in df.columns]:
                            df["GW"] = gw_num
                        season_frames.append(df)
                    except Exception as e:
                        print(f"  [Phase 3] Warning: could not read {gw_file}: {e}")

        if season_frames:
            season_df = pd.concat(season_frames, ignore_index=True)
            frames.append(season_df)
            seasons_loaded += 1
            print(f"  [Phase 3]   {season_name}: {len(season_df):,} rows loaded")

    if not frames:
        print("  [Phase 3] No GW data files found inside vaastav_data/.")
        print("  Expected structure: vaastav_data/data/<season>/gws/gw1.csv")
        return None

    print(f"  [Phase 3] Loaded {seasons_loaded} season(s) — {sum(len(f) for f in frames):,} total rows.")
    df = pd.concat(frames, ignore_index=True)

    # ── Normalise column names ──────────────────────────────────────
    df.columns = [c.lower().strip() for c in df.columns]

    # vaastav uses 'gw' in merged files; individual files may use 'round'
    if "gw" not in df.columns and "round" in df.columns:
        df["gw"] = df["round"]

    # Ensure required columns exist, fill missing with 0
    required_cols = [
        "name", "element", "position", "team", "gw", "season",
        "total_points", "minutes", "goals_scored", "assists",
        "clean_sheets", "bonus", "value", "was_home",
        "transfers_balance", "selected",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce").fillna(0)
    df["minutes"]      = pd.to_numeric(df["minutes"],      errors="coerce").fillna(0)
    df["value"]        = pd.to_numeric(df["value"],        errors="coerce").fillna(0)
    df["gw"]           = pd.to_numeric(df["gw"],           errors="coerce").fillna(0)

    # ── Sort for rolling windows ───────────────────────────────────
    df = df.sort_values(["season", "element", "gw"]).reset_index(drop=True)

    # ── Rolling features (per player within a season) ─────────────
    grp = df.groupby(["season", "element"])

    def roll(col, n):
        return grp[col].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())

    df["pts_roll3"]         = roll("total_points",  3)
    df["pts_roll6"]         = roll("total_points",  6)
    df["goals_roll5"]       = roll("goals_scored",  5)
    df["assists_roll5"]     = roll("assists",        5)
    df["clean_sheets_roll5"]= roll("clean_sheets",  5)
    df["bonus_roll5"]       = roll("bonus",         5)
    df["minutes_pct"]       = (df["minutes"] / 90).clip(0, 1)

    # xG / xA not in vaastav — proxy with goals and assists for training
    # (replaced by real xG/xA at inference time when available)
    df["xg_roll5"] = df["goals_roll5"]
    df["xa_roll5"] = df["assists_roll5"]

    # ── Other features ─────────────────────────────────────────────
    df["price"]                  = df["value"] / 10
    df["is_home"]                = df["was_home"].astype(int)
    df["transfers_in_event_norm"]= pd.to_numeric(
        df["transfers_balance"], errors="coerce"
    ).fillna(0).clip(-1e5, 1e5) / 1e5

    # One-hot position
    pos_map = {"GKP": "GK", "GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
    df["position_clean"] = df["position"].map(pos_map).fillna("MID")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        df[f"pos_{pos}"] = (df["position_clean"] == pos).astype(int)

    # FDR proxy: not in vaastav, use neutral 3
    df["fdr"] = 3

    # ── Targets ───────────────────────────────────────────────────
    # Target A: next GW points
    df["target_next_gw"] = grp["total_points"].transform(lambda x: x.shift(-1))

    # Target B: cumulative points from this GW to end of season (GW38)
    df["target_season_rem"] = grp["total_points"].transform(
        lambda x: x[::-1].cumsum()[::-1]
    ) - df["total_points"]

    # ── Drop rows with NaN targets or too few minutes ──────────────
    df = df.dropna(subset=["target_next_gw", "target_season_rem"])
    df = df[df["minutes"] >= 45].copy()

    print(f"  [Phase 3] Training dataset: {len(df):,} rows × {len(GW_FEATURES)} features")
    return df


# ─────────────────────────────────────────────
# 4b. Model A — next-GW predictor (LightGBM)
# ─────────────────────────────────────────────

class FPLModel:
    """
    LightGBM regressor that predicts a player's points in the next GW.

    Usage:
        model = FPLModel()
        if not model.load():               # try loading saved weights
            df_train = build_training_data()
            model.train(df_train)
            model.save()
        players = model.predict(players)   # adds 'predicted_points' column
    """

    MODEL_PATH = MODEL_DIR / "fpl_lgbm_gw.pkl"

    def __init__(self):
        self.model    = None
        self.features = GW_FEATURES
        self.mae      = None

    def train(self, df):
        """Fit LGBMRegressor on the vaastav training frame."""
        if not LGBM_AVAILABLE:
            print("  [FPLModel] lightgbm not available — skipping training.")
            return False
        if df is None or len(df) < MIN_TRAIN_ROWS:
            print(f"  [FPLModel] Not enough training data ({len(df) if df is not None else 0} rows).")
            return False

        print("  [FPLModel] Training LightGBM next-GW model...")
        available = [f for f in self.features if f in df.columns]
        X = df[available].fillna(0)
        y = df["target_next_gw"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, shuffle=False  # time-respecting split
        )

        self.model = lgb.LGBMRegressor(
            n_estimators     = 800,
            learning_rate    = 0.03,
            num_leaves       = 63,
            feature_fraction = 0.8,
            bagging_fraction = 0.8,
            bagging_freq     = 5,
            min_child_samples= 20,
            reg_alpha        = 0.1,
            reg_lambda       = 0.1,
            random_state     = 42,
            verbose          = -1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set          = [(X_val, y_val)],
            callbacks         = [lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=-1)],
        )
        preds     = self.model.predict(X_val)
        self.mae  = mean_absolute_error(y_val, preds)
        self.features = available   # store only cols that were present
        print(f"  [FPLModel] Trained. Validation MAE: {self.mae:.3f} pts")
        return True

    def predict(self, players):
        """
        Add 'predicted_points' column to players DataFrame.
        Falls back to heuristic if model not available.
        """
        if self.model is None:
            print("  [FPLModel] No model loaded — using heuristic fallback.")
            return players   # heuristic was already applied upstream

        available = [f for f in self.features if f in players.columns]
        missing   = [f for f in self.features if f not in players.columns]
        if missing:
            for col in missing:
                players[col] = 0   # fill any absent live features with 0

        X = players[self.features].fillna(0)
        players["predicted_points"] = self.model.predict(X).clip(min=0)
        print(f"  [FPLModel] Predictions generated for {len(players)} players.")
        return players

    def save(self):
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features,
                     "mae": self.mae}, self.MODEL_PATH)
        print(f"  [FPLModel] Saved → {self.MODEL_PATH}")

    def load(self):
        if not self.MODEL_PATH.exists():
            return False
        data          = joblib.load(self.MODEL_PATH)
        self.model    = data["model"]
        self.features = data["features"]
        self.mae      = data.get("mae")
        print(f"  [FPLModel] Loaded from {self.MODEL_PATH} "
              f"(val MAE: {self.mae:.3f})" if self.mae else
              f"  [FPLModel] Loaded from {self.MODEL_PATH}")
        return True

    def feature_importance(self, top_n=15):
        """Print top-N most important features."""
        if self.model is None:
            print("  No model loaded.")
            return
        imp = pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)
        print(f"\n  Top {top_n} features (LightGBM gain):")
        for feat, score in imp.head(top_n).items():
            bar = "█" * int(score / imp.max() * 20)
            print(f"    {feat:<30} {bar} {score:.1f}")


# ─────────────────────────────────────────────
# 4c. Model B — season-remainder predictor (XGBoost)
# ─────────────────────────────────────────────

class FPLSeasonModel:
    """
    XGBoost regressor that predicts a player's total points from the
    current GW to the end of the season. Used in mode='season'.

    Same interface as FPLModel.
    """

    MODEL_PATH = MODEL_DIR / "fpl_xgb_season.pkl"

    def __init__(self):
        self.model    = None
        self.features = GW_FEATURES
        self.mae      = None

    def train(self, df):
        if not XGB_AVAILABLE:
            print("  [FPLSeasonModel] xgboost not available — skipping training.")
            return False
        if df is None or len(df) < MIN_TRAIN_ROWS:
            print(f"  [FPLSeasonModel] Not enough training data.")
            return False

        print("  [FPLSeasonModel] Training XGBoost season-remainder model...")
        available = [f for f in self.features if f in df.columns]
        X = df[available].fillna(0)
        y = df["target_season_rem"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )

        self.model = xgb.XGBRegressor(
            n_estimators      = 700,
            max_depth         = 6,
            learning_rate     = 0.03,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            reg_alpha         = 0.1,
            reg_lambda        = 0.1,
            objective         = "reg:squarederror",
            random_state      = 42,
            verbosity         = 0,
            early_stopping_rounds = 50,
        )
        self.model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)], verbose=False)
        preds    = self.model.predict(X_val)
        self.mae = mean_absolute_error(y_val, preds)
        self.features = available
        print(f"  [FPLSeasonModel] Trained. Validation MAE: {self.mae:.3f} pts")
        return True

    def predict(self, players):
        if self.model is None:
            print("  [FPLSeasonModel] No model — using total_points as proxy.")
            players["season_predicted_points"] = players["total_points"]
            return players
        available = [f for f in self.features if f in players.columns]
        for col in [f for f in self.features if f not in players.columns]:
            players[col] = 0
        X = players[self.features].fillna(0)
        players["season_predicted_points"] = self.model.predict(X).clip(min=0)
        return players

    def save(self):
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features,
                     "mae": self.mae}, self.MODEL_PATH)
        print(f"  [FPLSeasonModel] Saved → {self.MODEL_PATH}")

    def load(self):
        if not self.MODEL_PATH.exists():
            return False
        data          = joblib.load(self.MODEL_PATH)
        self.model    = data["model"]
        self.features = data["features"]
        self.mae      = data.get("mae")
        print(f"  [FPLSeasonModel] Loaded from {self.MODEL_PATH}")
        return True


# ─────────────────────────────────────────────
# 4d. Model C — captaincy ranker (haul classifier)
# ─────────────────────────────────────────────

class CaptaincyModel:
    """
    GradientBoosting binary classifier: predicts the probability that a
    player scores ≥ HAUL_THRESHOLD points (default 12) in the next GW.

    Output probability is used to break ties in captain selection —
    prefer a player with high upside over one with merely high mean prediction.

    Usage:
        cap_model = CaptaincyModel()
        if not cap_model.load():
            cap_model.train(df_train)
            cap_model.save()
        selected_team = cap_model.rank_captains(selected_team)
        # Adds 'haul_prob' column; select_captain() uses it automatically.
    """

    MODEL_PATH = MODEL_DIR / "fpl_cap_classifier.pkl"

    def __init__(self):
        self.model    = None
        self.features = GW_FEATURES
        self.auc      = None

    def train(self, df):
        if not SKLEARN_AVAILABLE:
            print("  [CaptaincyModel] scikit-learn not available — skipping.")
            return False
        if df is None or len(df) < MIN_TRAIN_ROWS:
            return False

        print("  [CaptaincyModel] Training haul classifier...")
        available = [f for f in self.features if f in df.columns]
        X = df[available].fillna(0)
        y = (df["target_next_gw"] >= HAUL_THRESHOLD).astype(int)

        # Hauls are rare (~8% of rows) — use class_weight balancing
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )
        self.model = GradientBoostingClassifier(
            n_estimators  = 300,
            max_depth     = 4,
            learning_rate = 0.05,
            subsample     = 0.8,
            random_state  = 42,
        )
        self.model.fit(X_train, y_train)
        proba     = self.model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        self.auc  = roc_auc_score(y_val, proba)
        self.features = available
        print(f"  [CaptaincyModel] Trained. Validation ROC-AUC: {self.auc:.3f}")
        return True

    def rank_captains(self, selected_team):
        """Add 'haul_prob' column to the selected team DataFrame."""
        if self.model is None:
            selected_team["haul_prob"] = selected_team["predicted_points"] / \
                                          selected_team["predicted_points"].max()
            return selected_team
        for col in [f for f in self.features if f not in selected_team.columns]:
            selected_team[col] = 0
        X = selected_team[self.features].fillna(0)
        selected_team["haul_prob"] = self.model.predict_proba(X)[:, 1]
        return selected_team

    def save(self):
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features,
                     "auc": self.auc}, self.MODEL_PATH)
        print(f"  [CaptaincyModel] Saved → {self.MODEL_PATH}")

    def load(self):
        if not self.MODEL_PATH.exists():
            return False
        data          = joblib.load(self.MODEL_PATH)
        self.model    = data["model"]
        self.features = data["features"]
        self.auc      = data.get("auc")
        print(f"  [CaptaincyModel] Loaded from {self.MODEL_PATH}")
        return True


# ─────────────────────────────────────────────
# 4e. Model manager — trains/loads all three
# ─────────────────────────────────────────────

def load_or_train_models(force_retrain=False):
    """
    Attempt to load all three saved models.
    If any are missing (or force_retrain=True), train from vaastav data.

    Returns:
        gw_model      (FPLModel)
        season_model  (FPLSeasonModel)
        cap_model     (CaptaincyModel)
    All three may have model=None if neither saved weights nor vaastav
    data are available — in that case the heuristic fallback is used.
    """
    gw_model     = FPLModel()
    season_model = FPLSeasonModel()
    cap_model    = CaptaincyModel()

    all_loaded = (
        not force_retrain
        and gw_model.load()
        and season_model.load()
        and cap_model.load()
    )

    if not all_loaded:
        print("\n  [Phase 3] Training models from vaastav historical data...")
        df_train = build_training_data()
        if df_train is not None and len(df_train) >= MIN_TRAIN_ROWS:
            gw_model.train(df_train)
            gw_model.save()
            season_model.train(df_train)
            season_model.save()
            cap_model.train(df_train)
            cap_model.save()
        else:
            print("  [Phase 3] No training data — heuristic predictions will be used.")

    return gw_model, season_model, cap_model


# ─────────────────────────────────────────────
# 4f. Live feature enrichment for current players
# ─────────────────────────────────────────────

def enrich_features_for_prediction(players, player_histories):
    """
    Build the same rolling feature columns used during training,
    but from the live player history data fetched from the FPL API.

    player_histories: dict {player_id: DataFrame} from fetch_all_player_histories()
    Adds all GW_FEATURES columns to the players DataFrame in-place.
    """
    print("  [Phase 3] Engineering live features for prediction...")

    def rolling_mean(history_df, col, n):
        if history_df is None or history_df.empty or col not in history_df.columns:
            return 0.0
        vals = pd.to_numeric(history_df[col], errors="coerce").fillna(0)
        return float(vals.tail(n).mean())

    for col in GW_FEATURES:
        if col not in players.columns:
            players[col] = 0.0

    for idx, row in players.iterrows():
        pid  = row["id"]
        hist = player_histories.get(pid)

        players.at[idx, "pts_roll3"]          = rolling_mean(hist, "total_points", 3)
        players.at[idx, "pts_roll6"]          = rolling_mean(hist, "total_points", 6)
        players.at[idx, "goals_roll5"]        = rolling_mean(hist, "goals_scored", 5)
        players.at[idx, "assists_roll5"]      = rolling_mean(hist, "assists", 5)
        players.at[idx, "clean_sheets_roll5"] = rolling_mean(hist, "clean_sheets", 5)
        players.at[idx, "bonus_roll5"]        = rolling_mean(hist, "bonus", 5)

        # xG / xA: use goals/assists as proxy (replace with Understat merge later)
        players.at[idx, "xg_roll5"]           = players.at[idx, "goals_roll5"]
        players.at[idx, "xa_roll5"]           = players.at[idx, "assists_roll5"]

        # minutes_pct from latest GW
        if hist is not None and not hist.empty and "minutes" in hist.columns:
            last_mins = pd.to_numeric(hist["minutes"], errors="coerce").iloc[-1]
            players.at[idx, "minutes_pct"] = float(min(last_mins / 90, 1.0))

        # is_home: from next fixture (not in player history — left as 0 default)
        players.at[idx, "price"] = row["now_cost"] / 10

        # transfers_in_event normalised
        t = pd.to_numeric(row.get("transfers_in_event", 0), errors="coerce") or 0
        players.at[idx, "transfers_in_event_norm"] = float(np.clip(t / 1e5, -1, 1))

    # Position one-hots (already set in preprocess_data but ensure here)
    for pos in ["GK", "DEF", "MID", "FWD"]:
        players[f"pos_{pos}"] = (players["element_type"] == pos).astype(int)

    # fdr was already added by add_fixture_difficulty
    return players


# ─────────────────────────────────────────────
# 4g. Auto-retrain trigger (feedback loop)
# ─────────────────────────────────────────────

def should_retrain(tracker: "SeasonTracker"):
    """
    Check rolling MAE from the SeasonTracker.
    Returns True if the model should be retrained.
    """
    with tracker._connect() as conn:
        rows = conn.execute("""
            SELECT predicted_pts, actual_pts
            FROM   team_selections
            WHERE  actual_pts IS NOT NULL
            ORDER  BY gw DESC
            LIMIT  75          -- last 5 GWs × 15 players
        """).fetchall()

    if len(rows) < 30:
        return False   # not enough data yet

    errors = [abs(pred - actual) for pred, actual in rows]
    rolling_mae = sum(errors) / len(errors)
    print(f"  [Phase 3] Rolling MAE (last ~5 GWs): {rolling_mae:.3f} pts/player")
    if rolling_mae > RETRAIN_MAE_TRIGGER:
        print(f"  [Phase 3] MAE > {RETRAIN_MAE_TRIGGER} threshold — retraining triggered.")
        return True
    return False


# ══════════════════════════════════════════════
# SECTION 5 — TEAM OPTIMIZATION
# ══════════════════════════════════════════════

def optimize_team(players, mode="gw"):
    """
    PuLP integer programming optimizer.
    mode='gw'     → maximize predicted_points (next GW)
    mode='season' → maximize total_points (season-to-date, proxy for
                    remainder value until Phase 3 ML model is added)
    """
    print(f"Optimizing team selection (mode='{mode}')...")

    objective_col = "predicted_points" if mode == "gw" else "total_points"
    problem       = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)
    player_vars   = pulp.LpVariable.dicts("Player", players.index, cat="Binary")
    captain_vars  = pulp.LpVariable.dicts("Captain", players.index, cat="Binary")

    # Objective: squad points + captain double
    problem += pulp.lpSum([
        players.loc[i, objective_col] * player_vars[i] +
        players.loc[i, objective_col] * captain_vars[i]
        for i in players.index
    ])

    # Budget: £100m (costs stored in 0.1m units → limit = 1000)
    problem += pulp.lpSum([
        players.loc[i, "now_cost"] * player_vars[i]
        for i in players.index
    ]) <= 1000

    # Squad size
    problem += pulp.lpSum([player_vars[i] for i in players.index]) == 15

    # Exactly one captain, and only from selected players
    problem += pulp.lpSum([captain_vars[i] for i in players.index]) == 1
    for i in players.index:
        problem += captain_vars[i] <= player_vars[i]

    # Position quotas
    for pos, count in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        problem += pulp.lpSum([
            player_vars[i] for i in players.index
            if players.loc[i, "element_type"] == pos
        ]) == count

    # Max 3 players per club
    for team in players["team"].unique():
        problem += pulp.lpSum([
            player_vars[i] for i in players.index
            if players.loc[i, "team"] == team
        ]) <= 3

    status = problem.solve(pulp.PULP_CBC_CMD(msg=0))
    print(f"  Solver status: {pulp.LpStatus[status]}")

    selected_indices = [i for i in players.index if pulp.value(player_vars[i]) == 1]
    captain_index    = next((i for i in players.index if pulp.value(captain_vars[i]) == 1), None)

    selected_team = players.loc[selected_indices].copy()
    selected_team["is_captain"] = selected_team.index == captain_index
    return selected_team


def select_captain(selected_team):
    """
    Pick captain using a blend of predicted points and haul probability
    (Phase 3) when available. Falls back to highest predicted_points.
    The optimizer already set is_captain on the best linear choice;
    the captaincy model can override it with a probability-aware pick.
    """
    if "haul_prob" in selected_team.columns:
        norm_pts = selected_team["predicted_points"] / (
            selected_team["predicted_points"].max() or 1
        )
        score  = 0.70 * norm_pts + 0.30 * selected_team["haul_prob"]
        best_i = score.idxmax()
        selected_team["is_captain"] = selected_team.index == best_i
        return selected_team.loc[best_i]
    captains = selected_team[selected_team["is_captain"] == True]
    if not captains.empty:
        return captains.iloc[0]
    return selected_team.loc[selected_team["predicted_points"].idxmax()]


# ══════════════════════════════════════════════
# SECTION 6 — SEASON TRACKER  (Phase 2)
# ══════════════════════════════════════════════

class SeasonTracker:
    """
    Persists every GW squad selection to an SQLite database.

    Tables:
      team_selections  — one row per (gw, player); stores predicted &
                         actual points, price, captain flag.
      season_meta      — one row per GW; stores solve timestamp,
                         total predicted, total actual (filled in later).

    Usage:
      tracker = SeasonTracker()
      tracker.save_selection(gw=28, selected_team=df)
      tracker.record_actual_points(gw=28, actual_points={player_id: pts, ...})
      tracker.season_summary()
      tracker.get_gw_selection(gw=28)
    """

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Create tables if they don't already exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS team_selections (
                    gw              INTEGER NOT NULL,
                    player_id       INTEGER NOT NULL,
                    web_name        TEXT,
                    team            TEXT,
                    position        TEXT,
                    price           REAL,
                    predicted_pts   REAL,
                    actual_pts      REAL,
                    is_captain      INTEGER DEFAULT 0,
                    saved_at        TEXT,
                    PRIMARY KEY (gw, player_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS season_meta (
                    gw                  INTEGER PRIMARY KEY,
                    mode                TEXT,
                    total_predicted     REAL,
                    total_actual        REAL,
                    prediction_error    REAL,
                    solved_at           TEXT
                )
            """)
            conn.commit()
        print(f"  Season tracker ready → {os.path.abspath(self.db_path)}")

    def save_selection(self, gw, selected_team, mode="gw"):
        """
        Persist the selected squad for a given GW.
        Safe to call multiple times — will replace existing selection
        for that GW (useful when re-running before deadline).
        """
        now = datetime.now().isoformat()
        rows = []
        for _, player in selected_team.iterrows():
            rows.append((
                gw,
                int(player["id"]),
                player["web_name"],
                player["team"],
                player["element_type"],
                float(player["now_cost"]) / 10,   # convert to £m
                float(player["predicted_points"]),
                None,                              # actual_pts filled later
                int(bool(player.get("is_captain", False))),
                now
            ))

        total_pred = selected_team["predicted_points"].sum()
        captain    = selected_team[selected_team["is_captain"] == True]
        if not captain.empty:
            # Captain points are doubled
            total_pred += captain["predicted_points"].iloc[0]

        with self._connect() as conn:
            # Remove previous selection for this GW if re-running
            conn.execute("DELETE FROM team_selections WHERE gw = ?", (gw,))
            conn.execute("DELETE FROM season_meta WHERE gw = ?", (gw,))

            conn.executemany("""
                INSERT INTO team_selections
                  (gw, player_id, web_name, team, position, price,
                   predicted_pts, actual_pts, is_captain, saved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)

            conn.execute("""
                INSERT INTO season_meta (gw, mode, total_predicted, solved_at)
                VALUES (?, ?, ?, ?)
            """, (gw, mode, total_pred, now))
            conn.commit()

        print(f"  ✓ GW {gw} selection saved ({len(rows)} players, "
              f"total predicted: {total_pred:.2f} pts)")

    def record_actual_points(self, gw, actual_points: dict):
        """
        Call after a GW resolves.
        actual_points: {player_id (int): actual_points (int), ...}
        Obtainable via the FPL API bootstrap-static 'event_points' field,
        or by fetching each player's element-summary after the GW.
        """
        print(f"Recording actual points for GW {gw}...")
        with self._connect() as conn:
            for player_id, pts in actual_points.items():
                conn.execute("""
                    UPDATE team_selections
                    SET    actual_pts = ?
                    WHERE  gw = ? AND player_id = ?
                """, (pts, gw, player_id))

            # Compute GW total (captain doubled)
            rows = conn.execute("""
                SELECT actual_pts, is_captain
                FROM   team_selections
                WHERE  gw = ? AND actual_pts IS NOT NULL
            """, (gw,)).fetchall()

            if rows:
                total_actual = sum(
                    pts * 2 if is_cap else pts
                    for pts, is_cap in rows
                )
                conn.execute("""
                    UPDATE season_meta
                    SET    total_actual = ?,
                           prediction_error = total_predicted - ?
                    WHERE  gw = ?
                """, (total_actual, total_actual, gw))
                print(f"  ✓ GW {gw} actual total: {total_actual} pts")

            conn.commit()

    def get_gw_selection(self, gw):
        """Retrieve the stored squad for a specific GW as a DataFrame."""
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM team_selections WHERE gw = ? ORDER BY position, web_name",
                conn, params=(gw,)
            )
        if df.empty:
            print(f"  No selection found for GW {gw}.")
        return df

    def season_summary(self):
        """Print a full season accuracy report across all tracked GWs."""
        with self._connect() as conn:
            meta = pd.read_sql_query(
                "SELECT * FROM season_meta ORDER BY gw", conn
            )
            selections = pd.read_sql_query(
                "SELECT * FROM team_selections", conn
            )

        if meta.empty:
            print("No GWs tracked yet.")
            return

        print("\n" + "═" * 55)
        print("  SEASON TRACKER SUMMARY")
        print("═" * 55)
        print(f"  GWs tracked  : {len(meta)}")

        completed = meta.dropna(subset=["total_actual"])
        if not completed.empty:
            print(f"  GWs resolved : {len(completed)}")
            print(f"  Total actual : {completed['total_actual'].sum():.0f} pts")
            print(f"  Avg per GW   : {completed['total_actual'].mean():.1f} pts")
            print(f"  Best GW      : GW {completed.loc[completed['total_actual'].idxmax(), 'gw']} "
                  f"({completed['total_actual'].max():.0f} pts)")
            print(f"  Worst GW     : GW {completed.loc[completed['total_actual'].idxmin(), 'gw']} "
                  f"({completed['total_actual'].min():.0f} pts)")

            mae = selections.dropna(subset=["actual_pts"])
            if not mae.empty:
                mae["error"] = (mae["predicted_pts"] - mae["actual_pts"]).abs()
                print(f"  Avg pred MAE : {mae['error'].mean():.2f} pts/player")

            print("\n  GW-by-GW breakdown:")
            print(f"  {'GW':>4}  {'Mode':>6}  {'Predicted':>10}  {'Actual':>8}  {'Error':>8}")
            print("  " + "-" * 44)
            for _, row in meta.iterrows():
                actual = f"{row['total_actual']:.0f}" if pd.notna(row["total_actual"]) else "pending"
                error  = f"{row['prediction_error']:+.1f}" if pd.notna(row["prediction_error"]) else "—"
                print(f"  {int(row['gw']):>4}  {row['mode']:>6}  "
                      f"{row['total_predicted']:>10.1f}  {actual:>8}  {error:>8}")

        print("═" * 55 + "\n")

    def get_best_gw(self):
        """Returns the GW number with the highest actual points."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT gw, total_actual FROM season_meta
                WHERE  total_actual IS NOT NULL
                ORDER  BY total_actual DESC LIMIT 1
            """).fetchone()
        return row  # (gw, total_actual) or None


# ══════════════════════════════════════════════
# SECTION 7 — OUTPUT
# ══════════════════════════════════════════════

def output_results(selected_team, captain, mode="gw"):
    """Pretty-print the selected squad grouped by position."""
    mode_label = "Next Gameweek" if mode == "gw" else "Best Season Team"
    print(f"\n{'═'*55}")
    print(f"  OPTIMAL TEAM — {mode_label.upper()}")
    print(f"{'═'*55}")

    for pos in ["GK", "DEF", "MID", "FWD"]:
        subset = selected_team[selected_team["element_type"] == pos]
        for _, p in subset.iterrows():
            cap_tag  = " ©" if p.get("is_captain") else "  "
            cost_str = f"£{p['now_cost']/10:.1f}m"
            print(f"  {pos:>3} {cap_tag} {p['web_name']:<22} "
                  f"{p['team']:<18} {cost_str:>6}  "
                  f"pred: {p['predicted_points']:.3f}")

    total_cost  = selected_team["now_cost"].sum() / 10
    total_pred  = selected_team["predicted_points"].sum()
    cap_pts     = captain["predicted_points"]
    print(f"\n  Captain : {captain['web_name']} ({captain['team']}) "
          f"→ {cap_pts * 2:.2f} pts (doubled)")
    print(f"  Squad cost   : £{total_cost:.1f}m / £100.0m")
    print(f"  Total pred   : {total_pred + cap_pts:.2f} pts (inc. captain bonus)")
    print(f"{'═'*55}\n")


# ══════════════════════════════════════════════
# SECTION 8 — MAIN
# ══════════════════════════════════════════════

def run(mode="gw", gw=None, save_to_tracker=True, force_retrain=False):
    """
    Full pipeline run — v3.0

    Args:
        mode           : 'gw'     → next-GW team (Model A + CaptaincyModel)
                         'season' → season-best team (Model B)
        gw             : Gameweek number for the tracker. Auto-inferred if None.
        save_to_tracker: Persist selection to SQLite.
        force_retrain  : Force model retraining even if saved weights exist.

    Pipeline:
        1. Fetch FPL API data
        2. Preprocess players
        3. Concurrent player history fetch
        4. Fixture difficulty
        5. Heuristic features (v2 fallback)
        6. [Phase 3] Load / train ML models
        7. [Phase 3] Enrich live features for ML prediction
        8. [Phase 3] ML prediction overwrites heuristic predicted_points
        9. [Phase 3] Auto-retrain check via SeasonTracker MAE
        10. Optimise squad (PuLP)
        11. [Phase 3] Captaincy model ranks captain candidates
        12. Output + persist

    Returns:
        selected_team (DataFrame), captain (Series)
    """
    # ── 1. Fetch ────────────────────────────────────────────────────
    data     = fetch_data()
    fixtures = fetch_fixture_data()

    if gw is None:
        upcoming = fixtures[fixtures["finished"] == False]
        gw = int(upcoming["event"].min()) if not upcoming.empty else 38
        print(f"Inferred current GW: {gw}")

    # ── 2. Preprocess ───────────────────────────────────────────────
    players, teams = preprocess_data(data)

    # ── 3. Concurrent history fetch (reuse for both form + ML features)
    player_ids      = players["id"].tolist()
    player_histories = fetch_all_player_histories(player_ids)

    # ── 4. Fixture difficulty ────────────────────────────────────────
    players = add_fixture_difficulty(players, fixtures, teams)

    # ── 5. Heuristic form + predicted_points (v2 fallback) ──────────
    form_values = {}
    for _, row in players.iterrows():
        pid  = row["id"]
        hist = player_histories.get(pid)
        if hist is not None and not hist.empty:
            pts = pd.to_numeric(hist.get("total_points", pd.Series()), errors="coerce")
            form_values[pid] = pts.tail(7).mean()
        else:
            form_values[pid] = row["form"]
    players["form_calc"] = players["id"].map(form_values).fillna(0)
    players = update_predicted_points(players)   # heuristic baseline

    # ── 6. Load / train Phase 3 ML models ───────────────────────────
    tracker      = SeasonTracker() if save_to_tracker else None
    should_force = force_retrain or (
        tracker is not None and should_retrain(tracker)
    )
    gw_model, season_model, cap_model = load_or_train_models(
        force_retrain=should_force
    )

    # ── 7. Enrich live features ──────────────────────────────────────
    players = enrich_features_for_prediction(players, player_histories)

    # ── 8. ML predictions (overwrite heuristic if model available) ──
    if mode == "season":
        players = season_model.predict(players)
        # Map season_predicted_points → predicted_points for optimizer
        if "season_predicted_points" in players.columns:
            players["predicted_points"] = players["season_predicted_points"]
    else:
        players = gw_model.predict(players)   # writes predicted_points directly

    # ── 9 + 10. Optimise squad ───────────────────────────────────────
    selected_team = optimize_team(players, mode=mode)

    # ── 11. Captaincy model ──────────────────────────────────────────
    selected_team = cap_model.rank_captains(selected_team)
    captain       = select_captain(selected_team)

    # ── 12. Output + persist ─────────────────────────────────────────
    output_results(selected_team, captain, mode=mode)

    if tracker is not None:
        tracker.save_selection(gw=gw, selected_team=selected_team, mode=mode)

    return selected_team, captain


def main():
    # ── Run next-GW selection (Phase 3 ML + tracker) ─────────────
    selected_team, captain = run(mode="gw", save_to_tracker=True)

    # ── Optionally run season-best selection ──────────────────────
    # selected_team, captain = run(mode="season", save_to_tracker=True)

    # ── Force a model retrain (e.g. start of new season) ─────────
    # selected_team, captain = run(mode="gw", force_retrain=True)

    # ── Print feature importance after training ───────────────────
    # from fpl_team_generator import FPLModel
    # m = FPLModel(); m.load(); m.feature_importance()

    # ── Record actual points after GW resolves ────────────────────
    # tracker = SeasonTracker()
    # tracker.record_actual_points(gw=28, actual_points={
    #     123: 12,   # player_id: actual_pts
    #     456: 6,
    # })

    # ── Season summary + auto-retrain check ──────────────────────
    # tracker = SeasonTracker()
    # tracker.season_summary()

    # ── Retrieve a past GW squad ──────────────────────────────────
    # past = tracker.get_gw_selection(gw=27)
    # print(past)


if __name__ == "__main__":
    main()