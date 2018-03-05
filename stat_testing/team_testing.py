import pandas as pd
import numpy as np
import time
from scipy.stats.stats import pearsonr
from matplotlib import pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'


def get_season(date):
    """
    Get Season based on from_date

    :param date: date

    :return: season -> ex: 2016 for 2016-2017 season
    """
    year = date[:4]
    date = time.strptime(date, "%Y-%m-%d")

    if date > time.strptime('-'.join([year, '01-01']), "%Y-%m-%d"):
        if date < time.strptime('-'.join([year, '09-01']), "%Y-%m-%d"):
            return int(year) - 1
        else:
            return int(year)
    else:
        if date > time.strptime('-'.join([year, '07-01']), "%Y-%m-%d"):
            return int(year)
        else:
            return int(year) - 1


def fix_team(df):
    """
    Fix Teams (ATL/WPG and ARI/PHX)

    :param df: DataFrame 

    :return: With fixed column
    """
    df["Team"] = df.apply(lambda row: 'ARI' if row["Team"] == 'PHX' else row["Team"], axis=1)
    df["Team"] = df.apply(lambda row: 'WPG' if row["Team"] == 'ATL' else row["Team"], axis=1)

    return df


def get_team_stats(df):
    """
    Get Stats for team df

    :param df: Team DataFrame of raw numbers

    :return: Df with stats
    """
    for pre in ['G', 'C', 'F', 'xG', 'corsica_xG', 'sh_xG', 'wsh']:
        df['{}F%'.format(pre)] = df['{}F'.format(pre)] / (df['{}F'.format(pre)] + df['{}A'.format(pre)]) * 100
        df['{}F60'.format(pre)] = df['{}F'.format(pre)] * 60 / df['TOI']
        df['{}A60'.format(pre)] = df['{}A'.format(pre)] * 60 / df['TOI']

    return df


def team_preprocessing():
    """
    Pre-process Team info to get it how I want it
    """
    df_offside = pd.read_csv("data/OffsideReview_teams.csv")
    df_corsica = pd.read_csv("data/corsica_team_stats_adj.csv")

    # Fix up Seasons
    df_offside = df_offside[df_offside['Season'] != 2012]
    df_corsica['Season'] = df_corsica.apply(lambda row: get_season(row['Date']), axis=1)
    df_corsica = df_corsica[df_corsica['Season'] != 2012]

    # Fix Teams
    df_offside = fix_team(df_offside)
    df_corsica = fix_team(df_corsica)

    # Transfer stuff over from corsica to mine
    df_corsica = df_corsica[["Team", "Date", "xGF", "xGA"]]
    df = pd.merge(df_offside, df_corsica, on=["Team", "Date"], suffixes=['', '_corsica'])
    df['corsica_xGF'] = df['xGF_corsica']
    df['corsica_xGA'] = df['xGA_corsica']

    # Weighted Shots
    df['wshF'] = df['GF'] + .2 * (df['CF'] - df['GF'])
    df['wshA'] = df['GA'] + .2 * (df['CA'] - df['GA'])

    return df


def predict_team(df, col_dict, cols, k):
    """
    Predict Stats for Teams
    """
    col_start = 0
    for predict_col in ["GF%_n+1", "GF60_n+1", "GA60_n+1"]:
        preds = np.array(df[predict_col])
        for metric in cols[col_start: col_start + int(len(cols)/3)]:
            stat = np.array(df[metric])
            col_dict[metric][str(k)].append(np.arctanh(pearsonr(stat, preds)[0]))

        col_start += int(len(cols)/3)


def make_graph(samples, cors, legends, predict):
    fig = plt.figure()
    plt.title("Predicting Future {} for Teams".format(predict))
    plt.xlabel('Num Predictive Games')
    plt.ylabel('R^2')

    for i in range(len(samples)):
        plt.plot(samples[i], cors[i], label=legends[i])

    #plt.xlim([0, 80])
    plt.legend(loc=8)

    fig.savefig("Team_{}_foo_bar.png".format(predict))
    plt.close()


def get_cors(stat_cors):
    """
    Get Correlations for values of k for each metric
    """
    predict_cols = ["GF%", "GF60", "GA60"]

    # Samples hold the specific k, cors hold the correlation, and legends hold the stat name
    samples, cors, legends = [], [], []

    i = 0
    for col in stat_cors.keys():
        sample, cor = [], [],
        for k in stat_cors[col].keys():
            avg = sum(stat_cors[col][k]) / float(len(stat_cors[col][k]))
            print("Stat: {},".format(col), "K: {},".format(k), "r: {}".format(np.tanh(avg)))

            # r^2 not r
            cor.append(round(np.tanh(avg) ** 2, 3))
            sample.append(int(k))

        samples.append(sample)
        cors.append(cor)
        legends.append(col)

        i += 1
        print("\n")

        # 6, 12, 18 -> end of each predict metric
        if i % 6 == 0:
            make_graph(samples, cors, legends, predict_cols[int(i/6)-1])
            samples, cors, legends = [], [], []


def team_analysis(df, if_chron):
    """
    If 'if_chron' is true, then I do a Chronological in Season Prediction (see here: https://hockey-graphs.com/2014/11/13/adjusted-possession-measures/)
    for predictivity (of future goals). 
    
    If it isn't True, I do it like DTM did here (https://hockey-graphs.com/2015/10/01/expected-goals-are-a-better-predictor-of-future-scoring-than-corsi-goals/)

    1. Cumulative: GF%, FF%, CF%, xGF%, sh_xGF%
    2. {}F60: GF60, FF60, CF60, xGF60, sh_xGF60 
    2. {}A60: GA60, FA60, CA60, xGA60, sh_xGA60 

    :return: None
    """
    cols = [
        "GF%", "CF%", "xGF%", 'corsica_xGF%', 'sh_xGF%', 'wshF%',
        "GF60", "CF60", "xGF60", 'corsica_xGF60', 'sh_xGF60', 'wshF60',
        "GA60", "CA60", "xGA60", 'corsica_xGA60', 'sh_xGA60', 'wshA60',
    ]

    teams_list = list(set(df['Team'].tolist()))

    # Get Num games to start and how much to walk
    start = walk = 2 if if_chron else 5

    # Get how many trial, if chronological obviously only 1
    trials = 1 if if_chron else 250

    stat_cors = {}
    for col in cols:
        stat_cors[col] = {str(key): [] for key in range(start, 80, walk)}

    for trial in range(trials):
        print(trial)
        for k in range(start, 80, walk):
            df1, df2 = pd.DataFrame(), pd.DataFrame()

            for season in range(2007, 2017):
                if season != 2012:
                    df_season = df[df.Season == season]

                    if if_chron:
                        df_season = df_season.sort_values(["Date"])
                    else:
                        df_season = df_season.sample(frac=1).reset_index(drop=True)

                    for team in teams_list:
                        team_df = df_season[df_season.Team == team]

                        # Split into k and 82-k
                        df_first = team_df.head(k)
                        df_second = team_df.tail(team_df.shape[0] - k)

                        # Group up stats in each sample
                        df_first = df_first.groupby(["Team"], as_index=False).sum()
                        df_second = df_second.groupby(["Team"], as_index=False).sum()

                        # Fix Season
                        df_first.Season = season
                        df_second.Season = season

                        # Append to master only if enough games (only an issue with 2007 for Corsica)
                        if team_df.shape[0] >= 80:
                            df1 = df1.append(df_first)
                            df2 = df2.append(df_second)

            # Get Stats for each Df
            df1 = get_team_stats(df1)
            df2 = get_team_stats(df2)

            # Only essential columns
            df1 = df1[["Team", "Season"] + cols]
            df2 = df2[["Team", "Season"] + cols]

            # Merge and what not
            df_merged = pd.merge(df1, df2, how="left", on=["Team", "Season"], suffixes=['', '_n+1'])

            predict_team(df_merged, stat_cors, cols, k)

    get_cors(stat_cors)


def main():
    team_analysis(team_preprocessing(), False)


if __name__ == '__main__':
    main()


