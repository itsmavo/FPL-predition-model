#!/usr/bin/env python3
import requests
import pandas as pd
import pulp
import numpy as np

# Constants
FORM_WEIGHT = 0.28
TOTAL_POINTS_WEIGHT = 0.72

# Fetch Data from FPL API
def fetch_data():
    print("Fetching data from FPL API... ")
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()
    return data

def fetch_fixture_data():
    print("Fetching fixture data from FPL API... ")
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)
    fixtures = pd.DataFrame(response.json())
    return fixtures

def fetch_player_history(player_id):
    print(f"Fetching player history for player ID: {player_id}... ")
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    response = requests.get(url)
    history = pd.DataFrame(response.json()['history'])
    return history

# Preprocess Data
def preprocess_data(data):
    print("Preprocessing data... ")
    players = pd.DataFrame(data['elements'])
    teams = pd.DataFrame(data['teams'])

    #Selecting relevant columns
    players = players[['id', 'web_name', 'team','element_type','now_cost',
                       'total_points', 'minutes', 'goals_scored', 'assists',
                       'clean_sheets','saves','bonus']]
    
    #Map team IDs to team names
    team_id_to_name = dict(zip(teams['id'], teams['name']))
    players['team'] = players['team'].map(team_id_to_name)

    #Map position IDs to position names
    position_id_to_name = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD' }
    players['element_type'] = players['element_type'].map(position_id_to_name)

    #Calculate points per game (ppg)
    players['ppg'] = players['total_points'] / (players['minutes'] / 90)

    #Handles players with 0 mins played (to avoid division by zero)
    players['ppg'] = players['ppg'].replace([np.inf, -np.inf], np.nan).fillna(0) 

    #Filter out players with fewer than 5 games played
    players = players[players['minutes'] >= 450]

    #Use ppg as the predicted points for the next game week
    players['predicted_points'] = players['ppg']

    #Ensure no player costs 0.0
    players['now_cost'] = players['now_cost'].fillna(0)

    return players, teams

#Calculate player form
def calculate_form(players):
    print("Calculating player form...")
    #Get last 5 game weeks
    players['form'] = 0.0 #Initialize form column

    for index, player in players.iterrows():
        player_id = player['id']
        history = fetch_player_history(player_id)

        if not history.empty:
            #Calculate average points over last 7 game weeks
            history['total_points'] = pd.to_numeric(history['total_points'])
            players.at[index, 'form'] = history['total_points'].tail(7).mean()
   
    return players

def add_fixture_difficulty(players, fixtures, teams):
    print("Adding fixture difficulty...")
    #Map team IDs to team names
    team_id_to_name = dict(zip(teams['id'], teams['name']))
    fixtures['team_a'] = fixtures['team_a'].map(team_id_to_name)
    fixtures['team_h'] = fixtures['team_h'].map(team_id_to_name)

    #Create a dictionary of fixture difficulty for each team
    fdr_dict = {}
    for _, fixture in fixtures.iterrows():
        fdr_dict[fixture['team_a']] = fixture['team_h_difficulty']
        fdr_dict[fixture['team_h']] = fixture['team_a_difficulty']

    #Map fixture difficulty to players
    players['fdr'] = players['team'].map(fdr_dict)
    return players

# Normalize the form and fixture difficulty values
def normalize_metric(series):
    """Normalize a given metric to a value between 0 and 1."""
    return (series - series.min()) / (series.max() - series.min())

def update_predicted_points(players):
    print("Updating predicted points...")
    # Adjust predicted points based on form and fixture difficulty

    # Normalize form and total points
    players['form_normalized'] = normalize_metric(players['form'])
    players['total_points_normalized'] = normalize_metric(players['total_points'])

    #Combine form and total points using weights
    players['predicted_points'] = (FORM_WEIGHT * players['form_normalized'] +
                                      TOTAL_POINTS_WEIGHT * players['total_points_normalized'])

    players['predicted_points'] = players['form'] * (1 / players['fdr']) # Higher FDR means lower predicted points
    return players

# Optimize team selection
def optimize_team(players):
    print("Optimizing team selection...")
    #define problem
    problem = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)

    #define decision variables
    player_vars = pulp.LpVariable.dicts("Player", players.index, cat="Binary")

    #Objective: Maximize predicted points
    problem += pulp.lpSum([players.loc[i, 'predicted_points'] * player_vars[i] for i in players.index])

    #Constraints
    # Budget (100 million)
    problem += pulp.lpSum([players.loc[i, 'now_cost'] * player_vars[i]
                           for i in players.index]) <= 1000 # Costs are in 0.1m units
    
    #Squad size (15 players)
    problem += pulp.lpSum([player_vars[i] for i in players.index]) == 15

    #Position Constraints
    problem += pulp.lpSum([player_vars[i] for i in players.index
                           if players.loc[i,'element_type'] == 'GK']) == 2
    problem += pulp.lpSum([player_vars[i] for i in players.index
                           if players.loc[i,'element_type'] == 'DEF']) == 5
    problem += pulp.lpSum([player_vars[i] for i in players.index
                           if players.loc[i,'element_type'] == 'MID']) == 5
    problem += pulp.lpSum([player_vars[i] for i in players.index
                           if players.loc[i,'element_type'] == 'FWD']) == 3

    #Team Constraint
    for team in players['team'].unique():
        problem += pulp.lpSum([player_vars[i] for i in players.index
                               if players.loc[i, 'team'] == team]) <= 3

    #Solve problem
    problem.solve()

    #Extract selected team
    selected_team = players.loc[[i for i in players.index if 
                                 pulp.value(player_vars[i]) == 1]]
    return selected_team

# Select Captain
def select_captain(selected_team):
    captain = selected_team.loc[selected_team['predicted_points'].idxmax()]
    return captain


#Output Results
def output_results(selected_team, captain):
    print("\nOptimal Team for Next Game Week: ")
    print(selected_team[['web_name', 'team', 'element_type', 'now_cost',
                         'predicted_points']].to_string(index=False))
    print(f"\nCaptain: {captain['web_name']} ({captain['team']}) - Predicted Points: {captain['predicted_points'] * 2 }")
    


#Main 
def main():
    data = fetch_data()
    fixtures = fetch_fixture_data()
    #preprocess data
    players, teams = preprocess_data(data)
    players = calculate_form(players)
    players = add_fixture_difficulty(players, fixtures, teams)
    players = update_predicted_points(players)
    selected_team = optimize_team(players)
    captain = select_captain(selected_team)
    output_results(selected_team, captain)

if __name__ == "__main__":
    main()