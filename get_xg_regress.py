import pandas as pd
import numpy as np
import random
import itertools
from scipy.stats.stats import pearsonr
import clean_data_xg as cdx


def fisher_z_to_r(fisher_samples):
    """
    Convert the fisher z score for each group of sample to the r
    Need to average all of the score for that sample (ex: 25 shots) and then convert the avg to r with tanh (since 
    arctanh was used to convert from r to z)
    
    :param fisher_samples: Samples for pos/sample_amount
    
    :return: dict of average r for each sample for F/D for both stats
    """
    new_cors = {"F": {}, "D": {}}

    for pos in fisher_samples.keys():
        for sample in fisher_samples[pos].keys():
            new_cors[pos][sample] = {"num": fisher_samples[pos][sample]["num"]}
            for stat in ["xg", "sh"]:
                avg = sum(fisher_samples[pos][sample][stat]) / float(len(fisher_samples[pos][sample][stat]))
                new_cors[pos][sample][stat] = np.tanh(avg)

    return new_cors


def get_pos(players, pos):
    """
    Filter players for given position 
    
    :param players: full dict of players
    :param pos: F/D
    
    :return: dict of just F or D
    """
    foo = {}
    for player in players.keys():
        if players[player]["pos"] == pos:
            foo[player] = players[player]

    return foo


def get_players_k(k, players):
    """
    Get players with at least k shots
    If they have at least k then take a random sample from the total amount
    
    :param k: number of shots
    :param players: dict of players
    
    :return: dict of players with a random sample of k shots
    """
    
    cron_players = {}
    for player in players.keys():
        if len(players[player]['data']) >= k:
            cron_players[player] = {"pos": players[player]["pos"]}
            cron_players[player]['data'] = random.sample(players[player]['data'], k)

    return cron_players


def get_stats(player, xg_stats1, xg_stats2, sh_stats1, sh_stats2, k):
    """
    Makes the function "get_r" cleaner. Split the sample in two and sum up the stats for both halves
    """
    shots = [shot["sh"] for shot in player["data"]]
    xg = [shot["xg"] for shot in player["data"]]

    xg_stats1.append([sum(shots[:int(k / 2)]) / sum(xg[:int(k / 2)])])
    sh_stats1.append([sum(shots[:int(k / 2)]) / len(shots[:int(k / 2)])])
    xg_stats2.append([sum(shots[int(k / 2):]) / sum(xg[int(k / 2):])])
    sh_stats2.append([sum(shots[int(k / 2):]) / len(shots[int(k / 2):])])


def get_r(k, players):
    """
    Get the Correlation Coefficient (pearson r) for stats
    
    :param k: num shots
    :param players: dict of players
    
    :return: r for xg and Fsh%
    """
    xg_stats1, sh_stats1, xg_stats2, sh_stats2 = [], [], [], []
    [get_stats(players[player], xg_stats1, xg_stats2, sh_stats1, sh_stats2, k) for player in players.keys()]

    # Fix Data
    xg_stats1, sh_stats1 = np.array(xg_stats1), np.array(sh_stats1)

    # Score
    xg_r = pearsonr(xg_stats1, xg_stats2)[0][0]
    sh_r = pearsonr(sh_stats1, sh_stats2)[0][0]

    return xg_r, sh_r


def assign_shots_event(play, players):
    """
    Assign the shot info for the event to the player
    
    :param play: Given event
    :param players: dict of players
     
    :return: None
    """
    players[play['player_season']]["data"].append({"xg": play['xg'], "sh": 1 if play['Event'] == "GOAL" else 0})


def get_player_seasons(df):
    """
    Return a dict for every player_season. Each value is an empty list
    
    :param df: DataFrame of all Goals/Sogs/Misses
    :return: dict of players
    """
    df['player_season'] = df['p1_name'] + " " + df['season'].map(str)
    player_seasons = df['player_season'].tolist()

    # Get Position Info
    df['shooter_hand'], df["shooter_pos"] = cdx.get_shooter_info(df)
    df['if_forward'] = np.where(df['shooter_pos'].isin(["F"]), 1, 0)
    if_forward = df['if_forward'].tolist()

    players = [[player_seasons[i], if_forward[i]] for i in range(len(if_forward))]
    players = list(k for k, _ in itertools.groupby(players))

    players_dict = {}
    for player in players:
        # the "data" list takes a dict {"xg": , "sh": } for each entry
        if player[1] == 1:
            players_dict[player[0]] = {"pos": "F", "data": []}
        else:
            players_dict[player[0]] = {"pos": "D", "data": []}

    return players_dict


def split_sample(players):
    """
    Gert r=.5 by weighting split sample correlations
    """
    # Takes list -> [Sample, xg_r, sh_r]
    fisher_samples = {"F": {}, "D": {}}

    for trial in range(250):
        print("Trial:", trial)

        k = 50
        cron_players = get_players_k(k, players)
        while len(cron_players.keys()) >= 100:
            for pos in ["F", "D"]:
                pos_players = get_pos(cron_players, pos)
                if len(pos_players.keys()) >= 100:
                    xg_r, sh_r = get_r(k, pos_players)

                    try:
                        fisher_samples[pos][str(k/2)]["xg"].append(np.arctanh(xg_r))
                        fisher_samples[pos][str(k/2)]["sh"].append(np.arctanh(sh_r))
                    except KeyError:
                        fisher_samples[pos][str(k/2)] = {"xg": [], "sh": [], "num": len(pos_players.keys())}
                        fisher_samples[pos][str(k/2)]["xg"].append(np.arctanh(xg_r))
                        fisher_samples[pos][str(k/2)]["sh"].append(np.arctanh(sh_r))

                    #print("K: {} -> Really {}".format(k, k/2), "Pos: {}".format(pos),
                    #      "Players: {}".format(len(pos_players.keys())),
                    #      "xg_r: {}".format(xg_r), "sh_r: {}".format(sh_r))

            #print("\n")
            k += 10
            cron_players = get_players_k(k, players)

    # Get Amounts
    get_regress_sample(fisher_samples)


def get_regress_sample(samples):
    """
    "Average" out r's at every k value. 
    Since we don't reach r=.5, I use Spearman-Brown Prophecy formula to get it and I then take the weighted average 
    based on the number of players to get my "regress" amount
    """
    new_cors = fisher_z_to_r(samples)

    for pos in new_cors.keys():
        print(pos)
        xg_denom, sh_denom, xg_wsum, sh_wsum = 0, 0, 0, 0
        for sample in new_cors[pos].keys():
            print(float(sample), new_cors[pos][sample]["xg"], new_cors[pos][sample]["sh"], new_cors[pos][sample]["num"])
            if new_cors[pos][sample]["xg"] > 0:
                xg_denom += new_cors[pos][sample]["num"]
                xg_wsum += (float(sample) * ((1 - new_cors[pos][sample]["xg"]) / new_cors[pos][sample]["xg"])) \
                            * new_cors[pos][sample]["num"]

            if new_cors[pos][sample]["sh"] > 0:
                sh_denom += new_cors[pos][sample]["num"]
                sh_wsum += (float(sample) * ((1 - new_cors[pos][sample]["sh"]) / new_cors[pos][sample]["sh"])) \
                            * new_cors[pos][sample]["num"]

        print("Position: {}".format(pos), "xg:", xg_wsum / xg_denom, "sh:", sh_wsum / sh_denom)


def get_full_xg():
    """
    Gets the analysis started
    """
    df = pd.read_csv("full_0716_xg.csv", index_col=0)

    # Only Regular Season and shots
    df = df[df['Game_Id'] < 30000]

    # Assign shit
    players = get_player_seasons(df)
    print("Converting to dict")
    plays = df.to_dict("records")
    print("Assigning Shots")
    [assign_shots_event(play, players) for play in plays]
    print("Finished Assigning Shots\n")

    split_sample(players)


def main():
    get_full_xg()


if __name__ == '__main__':
    main()


