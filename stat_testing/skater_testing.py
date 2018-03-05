"""
Analyze and test Shooter_xG. 
All data used here is 5v5 and is from the 2007-2016 seasons (all non-xG data is score adjusted)
"""

import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
pd.options.mode.chained_assignment = None  # default='warn'


def goalie_preprocessing():
    """
    Transfer from raw Data to how I want it (have to group team by season)

    :return: DataFrame
    """
    df = pd.read_csv("/Users/Student/hockey/shooter_xg/stat_testing/data/OffsideReview_goalies.csv")

    # Group up stats
    df = df[['Player', 'Season', 'GP', 'Team', 'Strength', 'TOI', 'GA', 'SA', 'FA', 'xGA', 'sh_xGA']]
    df_teams = df.groupby(['Player', 'Season'], as_index=False)['Team'].sum()
    df_stats = df.groupby(['Player', 'Season'], as_index=False)['GP', 'TOI', 'GA', 'SA', 'FA', 'xGA', 'sh_xGA'].sum()

    # Move Team over
    df_teams = df_teams.sort_values(['Player', 'Season'])
    df_stats = df_stats.sort_values(['Player', 'Season'])
    df_stats['Team'] = df_teams['Team']
    df_stats = fix_team(df_stats)

    return df_stats


def skater_preprocessing():
    """
    Transfer from raw Data to how I want it (have to group team by season)
    
    :return: DataFrame
    """
    df_ind = pd.read_csv("/Users/Student/hockey/shooter_xg/stat_testing/data/OffsideReview_skater_ind.csv")
    df_ice = pd.read_csv("/Users/Student/hockey/shooter_xg/stat_testing/data/OffsideReview_skater_ice.csv")
    df_rel = pd.read_csv("/Users/Student/hockey/shooter_xg/stat_testing/data/OffsideReview_skater_rel.csv")

    # Transfer over individual
    df_ice['Goals'] = df_ind['Goals']
    df_ice['xGoals'] = df_ind['xGoals']
    df_ice['sh_xGoals'] = df_ind['sh_xGoals']
    df_ice['iFen'] = df_ind['iFen']
    df_ice['iCorsi'] = df_ind['iCorsi']

    # Transfer over from rel
    df_ice['sh_xGF_off'] = df_rel['sh_xGF_off']
    df_ice['sh_xGA_off'] = df_rel['sh_xGA_off']
    df_ice['xGF_off'] = df_rel['xGF_off']
    df_ice['xGA_off'] = df_rel['xGA_off']
    df_ice['GF_off'] = df_rel['GF_off']
    df_ice['GA_off'] = df_rel['GA_off']
    df_ice['CF_off'] = df_rel['CF_off']
    df_ice['CA_off'] = df_rel['CA_off']
    df_ice['FF_off'] = df_rel['FF_off']
    df_ice['FA_off'] = df_rel['FA_off']
    df_ice['toi_off'] = df_rel['toi_off']

    df_teams = df_ice.groupby(["Player", "Season", "Position"], as_index=False)['Team'].sum()
    df_stats = df_ice.groupby(['Player', 'Season', 'Position'], as_index=False)[
        'GP', 'TOI', 'Goals', 'xGoals', 'sh_xGoals', 'iCorsi', 'iFen', 'GF', 'GA', 'CF', 'CA', 'FF', 'FA', 'xGA', 'xGF',
        'sh_xGF', 'sh_xGA', 'sh_xGF_off', 'sh_xGA_off', 'xGA_off', 'xGF_off', 'GF_off', 'GA_off', 'CF_off', 'CA_off',
        'FF_off', 'FA_off', 'toi_off'].sum()

    # Transfer over team...I do it separately because if you try doing numbers with text it doesn't do text
    df_teams = df_teams.sort_values(['Player', 'Season', 'Position'])
    df_stats = df_stats.sort_values(['Player', 'Season', 'Position'])
    df_stats['Team'] = df_teams['Team']
    df_stats = fix_team(df_stats)

    # Fix Position
    df_stats['Position'] = np.where(df_stats['Position'] == "D", "D", "F")

    return df_stats


def fix_team(df):
    """
    Fix Teams (ATL/WPG and ARI/PHX)
    
    :param df: DataFrame 
    
    :return: With fixed column
    """
    df["Team"] = df.apply(lambda row: 'ARI' if row["Team"] == 'PHX' else row["Team"], axis=1)
    df["Team"] = df.apply(lambda row: 'WPG' if row["Team"] == 'ATL' else row["Team"], axis=1)

    return df


def repeatability(df, metrics):
    """
    Get reliability for indicated metrics
    """
    print("Getting Reliability:")

    for metric in metrics:
        stat = np.array(df[metric])
        stat_nxt_yr = np.array(df[metric+"_n+1"])
        print("Correlation for {}:".format(metric), round(pearsonr(stat, stat_nxt_yr)[0], 3))


def predict_stat(df, metric_cols, predict_col):
    """
    Predict a given stats with other stats. 
    Ex: Predict GF% with CF%, FF% ...etc.
    """
    print("\nGetting Predictivity for: {}".format(predict_col))

    preds = np.array(df[predict_col])
    for metric in metric_cols:
        stat = np.array(df[metric])
        print("Correlation for {}:".format(metric), round(pearsonr(stat, preds)[0], 3))


def get_rel(row, prefix):
    """
    Get Rel Stat
    """
    for_stat = prefix+"F"
    ag_stat = prefix+"A"

    return (row[for_stat]/(row[for_stat]+row[ag_stat])) - (row[for_stat+"_off"]/(row[for_stat+"_off"]+row[ag_stat+"_off"]))


def get_rel_60(row, prefix, if_for):
    """
    Get Rel for per 60 stats
    """
    stat = prefix + "F" if if_for else prefix + "A"

    per60 = row[stat] * 60 / row['TOI']
    off_per60 = row[stat+"_off"] * 60 / row['toi_off']

    return per60 - off_per60


def get_skater_stats(df, toi):
    """
    Get the stats for skaters....
    """
    # Filter TOI >= 400
    df = df[df.TOI >= toi]

    # Individual Stats
    df['G60'] = df.apply(lambda row: row['Goals'] * 60 / row['TOI'], axis=1)
    df['ixG60'] = df.apply(lambda row: row['xGoals'] * 60 / row['TOI'], axis=1)
    df['sh_ixG60'] = df.apply(lambda row: row['sh_xGoals'] * 60 / row['TOI'], axis=1)
    df['iCors60'] = df.apply(lambda row: row['iCorsi'] * 60 / row['TOI'], axis=1)
    df['iFen60'] = df.apply(lambda row: row['iFen'] * 60 / row['TOI'], axis=1)

    # Weighted Shots
    df['wshF'] = df['GF'] + .2 * (df['CF'] - df['GF'])
    df['wshA'] = df['GA'] + .2 * (df['CA'] - df['GA'])
    df['wshF_off'] = df['GF_off'] + .2 * (df['CF_off'] - df['GF_off'])
    df['wshA_off'] = df['GA_off'] + .2 * (df['CA_off'] - df['GA_off'])

    stats_pre = ['G', 'xG', 'F', 'C', 'sh_xG', 'wsh']
    # Global Rel Stats
    for col in stats_pre:
        df['rel_' + col + "F%"] = df.apply(lambda row: get_rel(row, col), axis=1)

    # Rel For/Against 60 Stats
    for col in stats_pre:
        df['rel_{}F60'.format(col)] = df.apply(lambda row: get_rel_60(row, col, True), axis=1)
        df['rel_{}A60'.format(col)] = df.apply(lambda row: get_rel_60(row, col, False), axis=1)

    # ish_xG numbers
    df['ish_xGF'] = df.apply(lambda row: row['xGF'] - row['xGoals'] + row['sh_xGoals'], axis=1)
    df['ish_xGF%'] = df.apply(lambda row: row['ish_xGF'] / (row['ish_xGF'] + row['sh_xGA']), axis=1)
    df['rel_ish_xGF%'] = df.apply(lambda row: row['ish_xGF%'] - (row['sh_xGF_off'] / (row['sh_xGF_off'] + row['sh_xGA_off'])), axis=1)
    df['rel_ish_xGF60'] = df.apply(lambda row: (row['ish_xGF'] * 60 / row['TOI']) - (row['sh_xGF_off'] * 60 / row['toi_off']), axis=1)

    return df


def get_next_yr(df, df2):
    """
    Get stats for next year for on player level
    """
    # Merge TODO: Teams filter...
    df["Season_n+1"] = df["Season"] + 1
    df_merged = pd.merge(df, df2, how="left", left_on=["Player", "Season_n+1"], right_on=["Player", "Season"], suffixes=['', '_n+1'])
    df_merged = df_merged[~df_merged['GP_n+1'].isnull()]

    return df_merged


def goalie_analysis(df):
    """
    Year over Year Correlations for:
    1. Adjusted FSv% (Actual FSv% - xFSv%) for both standard xG model and shooter_xG model
    
    :return: None
    """
    cols = ['sv%', 'miss%', 'adj_fsv%', 'adj_sh_fsv%', 'fsv%']

    # Filter -> 20 Games played
    df = df[df['GP'] >= 15]

    # Get Stats
    df['sv%'] = 1 - (df['GA'] / df['SA'])
    df['miss%'] = 1 - (df['SA']/df['FA'])
    df['fsv%'] = 1 - (df['GA']/df['FA'])
    df['adj_fsv%'] = df['fsv%'] - (1 - (df['xGA'] / df['FA']))
    df['adj_sh_fsv%'] = df['fsv%'] - (1 - (df['sh_xGA'] / df['FA']))

    # Get Essential columns
    df = df[["Player", "Season", "Team", "GP"] + cols]

    df_merged = get_next_yr(df, df)

    print(df_merged.shape)
    repeatability(df_merged, cols)
    predict_stat(df_merged, cols, "fsv%_n+1")


def skater_analysis(df):
    """
    Year over Year Correlations for both predictivity (of future goals) and reliability:
    1. Shooter Stats: G60, ixG/60, shooter_ixG/60, iCorsi/60, iFen/60
    2. Relative Stats: GF%, FF%, CF%, xGF%, sh_xGF%, ish_xGF%
    3. Relative For Stats: GF60, FF60, CF60, xGF60, sh_GF60, ish_xGF60 
    4. Relative Against Stats: GA60, FA60, CA60, xGA60, sh_GA60
    
    Note: For numbers 2-3, ish_xGF{} refers to using a blend of my standard xG model and the shooter xG model. It's a
    piecewise function that is defined as being my shooter xG model when the shot is taken by the given player and my
    standard xG model otherwise. 
    
    :return: None
    """
    cols = ['G60', 'ixG60', 'sh_ixG60', 'iCors60', 'iFen60',
            'rel_GF%', 'rel_CF%', 'rel_FF%', 'rel_xGF%', 'rel_sh_xGF%', 'rel_ish_xGF%', 'rel_wshF%',
            'rel_GF60', 'rel_CF60', 'rel_FF60', 'rel_xGF60', 'rel_sh_xGF60', 'rel_ish_xGF60', 'rel_wshF60',
            'rel_GA60', 'rel_CA60', 'rel_FA60', 'rel_xGA60', 'rel_sh_xGA60', 'rel_wshA60',
            ]
    toi_min = 400

    # Just get stats for every year...used for predicting
    predict_df = get_skater_stats(df, toi_min)

    for offset in range(0, 3):
        dfs = []
        for season in range(2007 + offset, 2016):
            df_seasons = df[(df['Season'] >= season-offset) & (df['Season'] <= season)]
            df_seasons = df_seasons.groupby(['Player', 'Position'], as_index=False).sum()
            df_seasons['Season'] = season
            df_seasons['Team'] = ''
            dfs.append(df_seasons)

        df1 = get_skater_stats(pd.concat(dfs), toi_min*(offset+1))

        # Only essential columns and merge
        df1 = df1[["Player", "Season", "Position", "Team", 'GP'] + cols]
        df_merged = get_next_yr(df1, predict_df)

        # For Forwards and Defensemen
        for pos in ["F", "D"]:
            df_pos = df_merged[df_merged["Position"] == pos]
            #repeatability(df_merged, cols)

            print("\nPosition: {},".format(pos), "Players: {},".format(df_pos.shape[0]), "Seasons: {}".format(offset+1))

            predict_stat(df_pos, cols[:5], "G60_n+1")
            predict_stat(df_pos, cols[5:12], "rel_GF%_n+1")
            predict_stat(df_pos, cols[12:19], "rel_GF60_n+1")
            predict_stat(df_pos, cols[19:], "rel_GA60_n+1")


def main():
    skater_analysis(skater_preprocessing())

if __name__ == '__main__':
    main()


