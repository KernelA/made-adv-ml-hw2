
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_player_skills(skill_encoder: OneHotEncoder, coefs: np.ndarray) -> pd.DataFrame:
    rows = []
    all_players_ids = skill_encoder.categories_[0]
    for player_id in all_players_ids:
        rows.append({"player_id": player_id, "skill": coefs[np.where(
            all_players_ids == player_id)[0][0]]})
    return pd.DataFrame.from_records(rows, index="player_id")


def player2ratings(players_id, player_ratings):
    ratings = []
    for player_id in players_id:
        try:
            ratings.append(player_ratings.loc[player_id, "skill"])
        except KeyError:
            pass
    ratings.sort(reverse=True)
    return tuple(ratings)


def rank_teams(teams: pd.DataFrame, player_skills: pd.DataFrame):
    ranking_teams = teams.copy()
    ranking_teams["player_skils"] = ranking_teams["members"].apply(
        lambda x: player2ratings(x, player_skills))
    ranking_teams.sort_values("player_skils", ascending=False, inplace=True)
    ranking_teams.drop("player_skils", axis="columns", inplace=True)
    return ranking_teams


def estimate_rank(team_res: pd.DataFrame, player_ratings: pd.DataFrame):
    kendall_values = []
    for tour_id, teams in team_res.groupby("tour_id"):
        new_teams = teams[["members", "tour_rating"]].copy()
        new_teams.reset_index(inplace=True)
        original_order = new_teams.index.to_numpy()
        new_teams = rank_teams(new_teams, player_ratings)
        rank_order = new_teams.index.to_numpy()
        kendall_values.append(stats.kendalltau(original_order, rank_order)[0])
    return np.nanmean(kendall_values)
