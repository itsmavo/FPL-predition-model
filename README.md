# Fantasy Premier League (FPL) Prediction Model

An end-to-end machine learning pipeline for Fantasy Premier League that generates
optimal squads, tracks your season performance, and improves its predictions
gameweek by gameweek.

---

## Overview
 
This tool connects to the official FPL API, fetches live player and fixture data,
runs it through a trained machine learning stack, and solves an integer programming
optimisation problem to pick the best valid 15-man squad under FPL's constraints.
 
Three modes are available depending on your objective:
 
| Flag | Mode | Best used for |
|------|------|---------------|
| `-g` | Next gameweek | Weekly team selection and transfers |
| `-s` | Season remainder | Wildcard planning, pre-season setup |
| `-c` | Current season only | Fast heuristic pick, early season use |
 
Every selection is persisted to a local SQLite database so you can track prediction
accuracy over the season, review past squads, and log actual points after each
gameweek resolves.
 
---
 
## How It Works
 
```
FPL API  ──►  Preprocess  ──►  Feature Engineering  ──►  ML Models  ──►  Optimiser  ──►  Output
                                                              │                              │
                                                         vaastav CSVs              Season Tracker DB
```
 
**Data layer** — The official FPL API is called on every run to get current player
stats, prices, ownership, and fixture data. Player histories are fetched
concurrently (up to 10 threads) to keep runtime reasonable.
 
**Feature engineering** — Raw stats are transformed into rolling windows (last 3,
5, and 6 gameweeks), fixture difficulty scores, position one-hots, and value
metrics like points-per-million.
 
**ML stack** — Three complementary models handle different tasks:
 
- **Model A** (LightGBM) — predicts each player's points in the *next* gameweek
- **Model B** (XGBoost) — predicts each player's *season remainder* total
- **Model C** (GradientBoosting) — classifies the probability of a player scoring
  12+ points ("haul probability"), used exclusively for captaincy selection
 
**Optimiser** — A PuLP integer programming solver selects the best valid 15-man
squad maximising predicted points subject to FPL's hard constraints: £100m budget,
2 GKs, 5 DEFs, 5 MIDs, 3 FWDs, max 3 players per club, one captain.
 
**Season tracker** — Every squad selection is saved to `fpl_tracker.db`. After
each gameweek resolves you log actual points back against predictions. Rolling
prediction error is monitored — if it drifts above 3.5 pts/player the system
triggers automatic model retraining.
 
---
 
## Project Structure
 
```
FPL-prediction-model/
├── fpl_team_generator.py   # Main script — all logic in one file
├── README.md
├── .gitignore
│
├── vaastav_data/           # ← NOT committed. Clone separately (see below)
│   └── data/
│       ├── 2018-19/gws/gw1.csv ... gw38.csv
│       ├── 2019-20/gws/gw1.csv ... gw38.csv
│       ├── ...
│       └── 2023-24/gws/merged_gw.csv
│
├── models/                 # ← NOT committed. Auto-created on first run
│   ├── fpl_lgbm_gw.pkl         # Model A weights
│   ├── fpl_xgb_season.pkl      # Model B weights
│   └── fpl_cap_classifier.pkl  # Model C weights
│
└── fpl_tracker.db          # ← NOT committed. Local season history
```
 
---
 
## Installation
 
### Requirements
 
- Python 3.8 or higher
- Internet access (FPL API is called on every run)
 
### 1. Clone the repository
 
```bash
git clone https://github.com/itsmavo/FPL-predition-model
cd FPL-predition-model
```
 
### 2. Create a virtual environment
 
```bash
python3 -m venv fpl_env
 
# macOS / Linux
source fpl_env/bin/activate
 
# Windows
fpl_env\Scripts\activate
```
 
### 3. Install dependencies
 
```bash
pip install requests pandas pulp numpy lightgbm xgboost scikit-learn joblib
```
 
Full dependency list:
 
| Package | Purpose |
|---------|---------|
| `requests` | FPL API HTTP calls |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `pulp` | Integer programming optimiser |
| `lightgbm` | Model A — next-GW prediction |
| `xgboost` | Model B — season remainder prediction |
| `scikit-learn` | Model C — captaincy classifier |
| `joblib` | Model serialisation (save/load) |
 
> `sqlite3`, `os`, `datetime`, `concurrent.futures`, `argparse`, `pathlib` are all
> part of Python's standard library — no installation needed.
 
---
 
## Getting the Training Data
 
The ML models train on the
[vaastav Fantasy Premier League dataset](https://github.com/vaastav/Fantasy-Premier-League),
which contains historical player stats for every gameweek going back to 2016-17.
 
Clone it into your project folder:
 
```bash
git clone https://github.com/vaastav/Fantasy-Premier-League vaastav_data
```
 
The script auto-detects the repo's layout. After cloning your structure should
look like:
 
```
FPL-prediction-model/
└── vaastav_data/
    └── data/
        ├── 2018-19/
        │   └── gws/
        │       ├── gw1.csv
        │       ├── gw2.csv
        │       └── ...
        ├── 2021-22/
        │   └── gws/
        │       └── merged_gw.csv
        └── ...
```
 
> The vaastav repo is ~500MB and is excluded from this repo's git history via
> `.gitignore`. You need to clone it separately on every machine you use.
 
---
 
## Quick Start
 
```bash
# Best team for next gameweek (trains models on first run)
python3 fpl-team-generator.py -g
 
# Best team for season remainder
python3 fpl-team-generator.py -s
 
# Fast heuristic pick — no ML training needed
python3 fpl-team-generator.py -c
```
 
**First run output** will look like this:
 
```
Fetching bootstrap data from FPL API...
Fetching fixture data from FPL API...
Preprocessing data...
Fetching player histories concurrently (550 players, up to 10 threads)...
  ...100/550 fetched
  ...200/550 fetched
  ...
 
  [Phase 3] Scanning vaastav_data/data for season folders...
  [Phase 3]   2018-19: 4,280 rows loaded
  [Phase 3]   2019-20: 4,010 rows loaded
  [Phase 3]   2020-21: 4,180 rows loaded
  [Phase 3]   2021-22: 3,890 rows loaded
  [Phase 3]   2022-23: 4,100 rows loaded
  [Phase 3]   2023-24: 3,950 rows loaded
  [Phase 3] Loaded 6 season(s) — 24,410 total rows.
 
  [FPLModel] Training LightGBM next-GW model...
  [FPLModel] Trained. Validation MAE: 2.41 pts
  [FPLSeasonModel] Training XGBoost season-remainder model...
  [FPLSeasonModel] Trained. Validation MAE: 18.3 pts
  [CaptaincyModel] Training haul classifier...
  [CaptaincyModel] Trained. Validation ROC-AUC: 0.74
 
═══════════════════════════════════════════════════════
  OPTIMAL TEAM — NEXT GAMEWEEK
═══════════════════════════════════════════════════════
   GK    Raya             Arsenal            £5.0m  pred: 5.21
   GK    Flekken          Brentford          £4.5m  pred: 3.84
  DEF  © Alexander-Arnold Liverpool          £8.5m  pred: 7.13
  DEF    Pedro Porro      Spurs              £5.5m  pred: 5.02
  ...
  Captain: Alexander-Arnold (Liverpool) → 14.26 pts (doubled)
  Squad cost  : £99.8m / £100.0m
  Total pred  : 87.43 pts (inc. captain bonus)
═══════════════════════════════════════════════════════
```
 
Subsequent runs skip training and load saved model weights in under a second.
 
---
 
## CLI Reference
 
```
python3 fpl-team-gen.py [MODE] [OPTIONS]
```
 
### Team Selection Modes
 
These flags are mutually exclusive — only one can be used at a time.
 
#### `-g` / `--gameweek` — Best team for next gameweek *(default)*
 
```bash
python3 fpl-team-gen.py -g
```
 
Uses **Model A** (LightGBM) trained on historical FPL data to predict each
player's points in the upcoming gameweek. Captaincy is decided by a blend of
predicted points (70%) and haul probability from **Model C** (30%), favouring
players with high upside over players with a merely consistent mean.
 
This is the mode you will use most often — run it before each gameweek deadline.
 
---
 
#### `-s` / `--season` — Best team for season remainder
 
```bash
python3 fpl-team-gen.py -s
```
 
Uses **Model B** (XGBoost) trained to predict each player's cumulative total
points from the current gameweek to GW38. The optimiser then picks the squad
that maximises that figure rather than just next week's return.
 
Best used when planning a wildcard, free hit, or building a pre-season squad
where long-term value matters more than immediate returns.
 
---
 
#### `-c` / `--current` — Current season stats only
 
```bash
python3 fpl-team-gen.py -c
```
 
Bypasses all ML models entirely. Uses only live FPL API data — points-per-game
(PPG), FPL's own rolling form score, and fixture difficulty rating — to pick the
team. No vaastav data needed, no training time.
 
Best used:
- At the very start of a season when there is not yet enough data to train on
- When you want a quick sanity-check pick without waiting for the ML pipeline
- When vaastav data is unavailable
 
---
 
### Tracker Commands
 
These commands do not generate a team — they query or update the season history
database only.
 
#### `--summary` — Season accuracy report
 
```bash
python3 fpl-team-gen.py --summary
```
 
Prints a full breakdown of every gameweek tracked so far:
 
```
═══════════════════════════════════════════════════════
  SEASON TRACKER SUMMARY
═══════════════════════════════════════════════════════
  GWs tracked  : 12
  GWs resolved : 11
  Total actual : 743 pts
  Avg per GW   : 67.5 pts
  Best GW      : GW 8 (91 pts)
  Worst GW     : GW 3 (44 pts)
  Avg pred MAE : 2.87 pts/player
 
  GW-by-GW breakdown:
    GW    Mode   Predicted    Actual     Error
  --------------------------------------------
     1      gw        71.2        68      +3.2
     2      gw        68.4        74      -5.6
     ...
    12      gw        74.1   pending         —
═══════════════════════════════════════════════════════
```
 
---
 
#### `--history GW` — View a past gameweek's squad
 
```bash
python3 fpl-team-gen.py --history 8
```
 
Prints the 15-man squad selected in gameweek 8, including predicted points,
actual points (if recorded), price at the time, and captain flag.
 
---
 
#### `--record GW player_id:pts ...` — Log actual points
 
```bash
python3 fpl-team-gen.py --record 28 123:12 456:6 789:2 321:9
```
 
Call this after a gameweek finishes to record actual FPL points against each
player in your saved squad. The first argument is the gameweek number, followed
by any number of `player_id:actual_points` pairs.
 
Player IDs are shown when you run `--history` and are stored in `fpl_tracker.db`.
You can also find them in the FPL API response or on sites like FPLReview.
 
After recording, the season summary is printed automatically so you can see the
updated accuracy stats.
 
> **When to run this:** after the gameweek bonus points have been confirmed
> (usually Monday). Running it before bonuses are applied means your actual
> totals will be slightly off.
 
---
 
### ML Options
 
These flags modify how the ML pipeline behaves and can be combined with any
team selection mode.
 
#### `--retrain` — Force model retraining
 
```bash
python3 fpl-team-gen.py -g --retrain
```
 
Deletes and rebuilds all three saved model files from scratch using the vaastav
dataset. Use this:
- At the start of a new season after updating vaastav data
- If you have added new seasons to vaastav_data manually
- After the auto-retrain trigger fires and you want to do it manually
 
Without this flag, the script loads saved weights from the `models/` folder and
skips training entirely.
 
---
 
#### `--importance` — Feature importance table
 
```bash
python3 fpl-team-gen.py --importance
```
 
Prints the top 17 most predictive features from Model A (LightGBM), ranked by
gain score:
 
```
  Top 17 features (LightGBM gain):
    pts_roll3                      ████████████████████ 1847.3
    pts_roll6                      ██████████████       1201.4
    xg_roll5                       ████████████         987.2
    fdr                            █████████            712.8
    minutes_pct                    ███████              601.1
    ...
```
 
Useful for understanding what the model is actually learning and diagnosing
poor predictions.
 
---
 
#### `--gw N` — Override gameweek number
 
```bash
python3 fpl-team-gen.py -g --gw 32
```
 
By default the script infers the current gameweek from the next unplayed fixture.
Use `--gw` to override this — useful if the API returns a stale value near a
deadline or if you want to label a run with a specific GW number in the tracker.
 
---
 
#### `--no-save` — Run without saving to tracker
 
```bash
python3 fpl-team-gen.py -g --no-save
```
 
Generates and prints the team without writing anything to `fpl_tracker.db`.
Useful for exploratory runs, testing, or when you are not yet committed to
the selection.
 
---
 
## Season Workflow
 
Here is the recommended weekly routine:
 
### Before each gameweek deadline
 
```bash
# Generate your team for the upcoming GW
python3 fpl-team-gen.py -g
```
 
Review the output, make your transfers on the FPL website, then re-run if needed.
 
### After the gameweek resolves (Monday/Tuesday)
 
```bash
# Log actual points — get player IDs from --history or fpl_tracker.db
python3 fpl-team-gen.py --record 28 \
  123:12 456:6 789:2 321:9 654:6 \
  987:8 147:2 258:3 369:6 741:1 \
  852:9 963:6 174:2 285:6 396:3
```
 
### Anytime — check your season stats
 
```bash
python3 fpl-team-gen.py --summary
```
 
### Wildcard / free hit planning
 
```bash
# See what the model thinks is the best squad for the rest of the season
python3 fpl-team-gen.py -s
```
 
### Start of new season — update data and retrain
 
```bash
# Pull new vaastav season data
cd vaastav_data && git pull && cd ..
 
# Force retrain all three models
python3 fpl-team-gen.py -g --retrain
```
 
---
 
## How the Models Work
 
### Model A — LightGBM (next-GW prediction)
 
Trained to predict the number of FPL points a player will score in the next
gameweek. Uses 17 features:
 
| Feature | Description |
|---------|-------------|
| `pts_roll3` | Average points over last 3 GWs |
| `pts_roll6` | Average points over last 6 GWs |
| `xg_roll5` | Expected goals rolling average (last 5 GWs) |
| `xa_roll5` | Expected assists rolling average (last 5 GWs) |
| `goals_roll5` | Goals scored rolling average |
| `assists_roll5` | Assists rolling average |
| `clean_sheets_roll5` | Clean sheets rolling average |
| `bonus_roll5` | Bonus points rolling average |
| `minutes_pct` | Minutes played as fraction of 90 |
| `fdr` | Fixture difficulty rating (1=easy, 5=hard) |
| `is_home` | 1 if playing at home |
| `pos_GK/DEF/MID/FWD` | Position one-hot encoding |
| `price` | Current price in £m |
| `transfers_in_event_norm` | Transfer activity (normalised) |
 
Trained with early stopping on a 15% validation split to prevent overfitting.
Saved to `models/fpl_lgbm_gw.pkl`.
 
### Model B — XGBoost (season remainder)
 
Same 17 features, different target: cumulative points from the current GW to GW38.
Trained on the full vaastav historical dataset with the label computed as a
reverse cumulative sum of each player's remaining gameweek scores within a season.
Saved to `models/fpl_xgb_season.pkl`.
 
### Model C — GradientBoosting Classifier (captaincy)
 
Binary classifier: does this player score ≥12 points this gameweek? The output
probability ("haul probability") is blended with Model A's raw prediction to pick
the captain — 70% mean prediction, 30% haul probability. This favours players with
genuine upside (attackers with easy home fixtures) over players who are merely
consistent. Saved to `models/fpl_cap_classifier.pkl`.
 
### Auto-retraining
 
After each gameweek where actual points are recorded, the rolling mean absolute
error across the last 5 GWs (~75 player-GW observations) is checked. If it
exceeds **3.5 pts/player**, the next run automatically triggers a full retrain
from vaastav data. The threshold can be adjusted via `RETRAIN_MAE_TRIGGER` in
the constants section at the top of the script.
 
### Heuristic fallback
 
If vaastav data is absent and no saved models exist, the system falls back to a
weighted heuristic: `predicted_points = (0.28 × form + 0.72 × total_points) × fdr_factor`.
No crash, no error — just a note in the output that heuristic predictions are
being used.
 
---
 
## Configuration
 
All tuneable parameters are constants at the top of `fpl_team_generator.py`:
 
```python
FORM_WEIGHT         = 0.28    # weight of recent form in heuristic fallback
TOTAL_POINTS_WEIGHT = 0.72    # weight of season total in heuristic fallback
FDR_WEIGHT          = 0.15    # how much fixture difficulty penalises predictions
MIN_MINUTES         = 450     # minimum minutes to be considered (~5 full games)
MAX_FORM_WORKERS    = 10      # parallel threads for player history fetching
DB_PATH             = "fpl_tracker.db"   # season tracker database location
MODEL_DIR           = Path("models")     # where trained models are saved
VAASTAV_DIR         = Path("vaastav_data")  # vaastav dataset root
HAUL_THRESHOLD      = 12      # pts — what counts as a haul for captaincy model
RETRAIN_MAE_TRIGGER = 3.5     # pts/player — auto-retrain threshold
MIN_TRAIN_ROWS      = 500     # minimum rows to attempt training
```
 
---
