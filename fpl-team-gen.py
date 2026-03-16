#!/usr/bin/env python3
"""
FPL Team Generator — v2.0
Fixes applied:
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
"""

import requests
import pandas as pd
import pulp
import numpy as np
import sqlite3
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
FORM_WEIGHT         = 0.28
TOTAL_POINTS_WEIGHT = 0.72
FDR_WEIGHT          = 0.15          # penalty applied on top of weighted score
MIN_MINUTES         = 450           # ~5 full games
MAX_FORM_WORKERS    = 10            # concurrent API requests for player history
DB_PATH             = "fpl_tracker.db"


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
# SECTION 4 — TEAM OPTIMIZATION
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
    """Returns the captain row (is_captain == True)."""
    captains = selected_team[selected_team["is_captain"] == True]
    if not captains.empty:
        return captains.iloc[0]
    # Fallback: highest predicted points
    return selected_team.loc[selected_team["predicted_points"].idxmax()]


# ══════════════════════════════════════════════
# SECTION 5 — SEASON TRACKER  (Phase 2)
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
# SECTION 6 — OUTPUT
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
# SECTION 7 — MAIN
# ══════════════════════════════════════════════

def run(mode="gw", gw=None, save_to_tracker=True):
    """
    Full pipeline run.

    Args:
        mode          : 'gw' for next-GW selection, 'season' for
                        season-best selection.
        gw            : Gameweek number to label this run in the tracker.
                        If None, it's inferred from the next unplayed fixture.
        save_to_tracker: Whether to persist the selection to SQLite.

    Returns:
        selected_team (DataFrame), captain (Series)
    """
    # --- Fetch ---
    data     = fetch_data()
    fixtures = fetch_fixture_data()

    # Infer current GW if not provided
    if gw is None:
        upcoming = fixtures[fixtures["finished"] == False]
        gw = int(upcoming["event"].min()) if not upcoming.empty else 38
        print(f"Inferred current GW: {gw}")

    # --- Preprocess ---
    players, teams = preprocess_data(data)

    # --- Features ---
    players = calculate_form(players)
    players = add_fixture_difficulty(players, fixtures, teams)
    players = update_predicted_points(players)

    # --- Optimise ---
    selected_team = optimize_team(players, mode=mode)
    captain       = select_captain(selected_team)

    # --- Output ---
    output_results(selected_team, captain, mode=mode)

    # --- Persist to Season Tracker ---
    if save_to_tracker:
        tracker = SeasonTracker()
        tracker.save_selection(gw=gw, selected_team=selected_team, mode=mode)

    return selected_team, captain


def main():
    # ── Run GW selection ──────────────────────────────────────────
    selected_team, captain = run(mode="gw", save_to_tracker=True)

    # ── Example: record actual points after GW resolves ──────────
    # Uncomment and populate after real GW results come in:
    #
    # tracker = SeasonTracker()
    # tracker.record_actual_points(gw=28, actual_points={
    #     123: 12,   # player_id: actual_pts
    #     456: 6,
    #     789: 2,
    #     # ... all 15 players
    # })

    # ── Example: view season summary ─────────────────────────────
    # tracker = SeasonTracker()
    # tracker.season_summary()

    # ── Example: retrieve a past GW squad ────────────────────────
    # past_squad = tracker.get_gw_selection(gw=27)
    # print(past_squad)


if __name__ == "__main__":
    main()