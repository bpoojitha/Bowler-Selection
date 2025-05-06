# Add data loading and feature engineering functions here
import pandas as pd
import app.services.datastore as datastore
import numpy as np


# Define match phases
def assign_phase(over):
    if over <= 6:
        return 'Powerplay'
    elif over <= 15:
        return 'MiddleOvers'
    else:
        return 'DeathOvers'


def aggregate_stats():
    # Merge 1: Merge bbb_df with mi_df
    datastore.merged_df = pd.merge(datastore.bbb_df, datastore.mi_df[datastore.mi_df['match_date'] >= '2020-01-01'],
                                   left_on='ID', right_on='match_number',
                                   how='left')

    datastore.merged_df['Phase'] = datastore.merged_df['Overs'].apply(assign_phase)

    # Calculate datastore.performance_df metrics
    datastore.performance_df = datastore.merged_df.groupby(
        ['Bowler', 'venue', 'BattingTeam', 'Phase', 'Batter', 'Innings', 'winner']).agg(
        TotalRuns=('TotalRun', 'sum'),
        ExtrasRuns=('ExtrasRun', 'sum'),
        WicketDeliveries=(
            'IsWicketDelivery',
            lambda x: np.sum(np.where(datastore.merged_df.loc[x.index, 'Kind'] != 'run out', x, 0))),
        Balls=('BallNumber', 'count'),
        Boundaries=('TotalRun', lambda x: (x >= 4).sum())  # Count of TotalRun >= 4
    ).reset_index()

    datastore.performance_df['WinLoss'] = np.where(
        datastore.performance_df['BattingTeam'] == datastore.performance_df['winner'], 0, 1)

    # Step 2: Merge the result with pi_df
    expanded_batter_df = pd.merge(datastore.performance_df, datastore.pi_df,
                                  left_on='Batter', right_on='battingName',
                                  how='left')

    expanded_batter_df.drop(
        columns=['ID', 'Name', 'longName', 'battingName', 'fieldingName', 'imgUrl', 'dob', 'longBattingStyles',
                 'bowlingStyles', 'longBowlingStyles', 'playingRoles', 'espn_url'], inplace=True)

    datastore.final_df = pd.merge(expanded_batter_df, datastore.pi_df,
                                  left_on='Bowler', right_on='Name',
                                  how='left')

    datastore.final_df.drop(
        columns=['ID', 'Name', 'longName', 'battingName', 'fieldingName', 'imgUrl', 'dob', 'battingStyles_y',
                 'longBattingStyles', 'playingRoles', 'espn_url'], inplace=True)

    datastore.final_df['EconomyRate'] = datastore.final_df['TotalRuns'] / (
            datastore.final_df['Balls'] / 6)

    datastore.final_df['StrikeRate'] = datastore.final_df['Balls'] / datastore.final_df[
        'WicketDeliveries']

    datastore.final_df['BoundaryPercentage'] = datastore.final_df['Boundaries'] / datastore.final_df[
        'Balls'] * 100

    datastore.final_df['Selected'] = 1

    datastore.final_df.to_csv('data/ipl/cleaned_data.csv', index=False, sep=',', header=True, encoding='utf-8')
