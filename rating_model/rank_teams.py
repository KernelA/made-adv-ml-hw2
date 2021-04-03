
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import random


def get_player_skills(skill_encoder: OneHotEncoder, coefs: np.ndarray) -> pd.DataFrame:
    """coefs contains a skill of players in the begin and follow complexity of the questions
    """
    player_ids = skill_encoder.categories_[0]
    return pd.DataFrame({"skill": coefs[:len(player_ids)]}, index=player_ids)


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
    ranking_teams["player_skills"] = ranking_teams["members"].apply(
        lambda x: player2ratings(x, player_skills))
    ranking_teams.sort_values("player_skills", ascending=False, inplace=True)
    ranking_teams.drop("player_skills", axis="columns", inplace=True)
    return ranking_teams


def estimate_rank(team_res: pd.DataFrame, player_ratings: pd.DataFrame):
    kendall_values = []
    spearmen_values = []
    for tour_id, teams in team_res.groupby("tour_id"):
        new_teams = teams[["members", "tour_rating"]].copy()
        new_teams.reset_index(inplace=True)
        original_order = new_teams.index.to_numpy()
        new_teams = rank_teams(new_teams, player_ratings)
        rank_order = new_teams.index.to_numpy()
        kendall_values.append(stats.kendalltau(original_order, rank_order)[0])
        spearmen_values.append(stats.spearmanr(original_order, rank_order)[0])
    return {"Kendall": np.nanmean(kendall_values), "Spearman": np.nanmean(spearmen_values)}
