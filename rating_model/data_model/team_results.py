import pickle
import warnings
from operator import attrgetter
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple
import logging

import tqdm
import pandas as pd

KEY_PATH = "key_path"

LOGGER = logging.getLogger("rating_model.model")


def init_from_data(cls, data: dict):
    kv_pairs = {}

    for field in fields(cls):
        if KEY_PATH in field.metadata:
            key_path = list(field.metadata[KEY_PATH])
            try:
                data_value = data[key_path.pop(0)]

                while key_path:
                    data_value = data_value[key_path.pop(0)]
            except KeyError:
                data_value = None
                LOGGER.exception("Detect None value in path.")

            kv_pairs[field.name] = data_value

    return cls(**kv_pairs)


@dataclass
class Player:
    player_id: int = field(metadata={KEY_PATH: ("player", "id")})
    rating: int = field(metadata={KEY_PATH: ("rating",)})
    used_rating: int = field(metadata={KEY_PATH: ("usedRating", )})

    def __post_init__(self):
        assert self.player_id >= 0, "Player id is negative"

    @ classmethod
    def from_dict(cls, data: dict) -> "Player":
        return init_from_data(cls, data)


@ dataclass
class Team:
    team_id: int = field(metadata={KEY_PATH: ("team", "id")})
    name: str = field(metadata={KEY_PATH: ("team", "name")})
    mask: Optional[Tuple[Optional[bool]]] = field(metadata={KEY_PATH: ("mask",)})
    position: Optional[int] = field(metadata={KEY_PATH: ("position",)})
    members: List[Player] = field(default_factory=list, init=False)

    def __post_init__(self):
        assert self.team_id >= 0,  "Team id is negative"
        assert self.position is None or self.position >= 0, "Position is negative"

        if self.mask is not None:
            new_values = []
            for value in self.mask:
                try:
                    new_values.append(bool(int(value)))
                except ValueError as exc:
                    warnings.warn(str(exc))
                    new_values.append(None)
            self.mask = tuple(new_values)

    @ classmethod
    def from_dict(cls, data: dict) -> "Team":
        team = init_from_data(cls, data)

        members = data["teamMembers"]

        for member_data in members:
            team.add_member(Player.from_dict(member_data))

        return team

    def add_member(self, member: Player):
        self.members.append(member)


@ dataclass
class Teams:
    # Key is team_id
    teams: Dict[int, Team] = field(init=False, default_factory=dict)

    def add_team(self, team: Team):
        if team.team_id in self.teams:
            raise ValueError("Team already exist")
        self.teams[team.team_id] = team

    @ classmethod
    def from_dict(cls, data: List[dict]) -> "Teams":
        teams = cls()

        for team_data in data:
            team = Team.from_dict(team_data)
            teams.add_team(team)

        return teams

    def __len__(self):
        return len(self.teams)

    def __iter__(self):
        return iter(self.teams)

    def __getitem__(self, team_id: int):
        return self.teams[team_id]


@ dataclass
class TeamResults:
    # Key is tournament_id
    tours: Dict[int, Teams] = field(init=False, default_factory=dict)

    @ staticmethod
    def load_pickle(file_obj) -> "TeamResults":
        data = pickle.load(file_obj)

        results = TeamResults()

        for tour_id in tqdm.tqdm(data, total=len(data)):
            if data[tour_id]:
                teams = Teams.from_dict(data[tour_id])
                results.add_result(tour_id, teams)

        return results

    def add_result(self, result_id: int, teams: Teams):
        self.tours[result_id] = teams

    def __len__(self):
        return len(self.tours)

    def __iter__(self):
        return iter(self.tours)

    def __getitem__(self, result_id: int) -> Teams:
        return self.tours[result_id]

    def filter_incorrect_questions_tours(self):
        total_none_questions = 0
        total_questions = 0
        total_deleted_teams = 0
        total_teams = 0

        for tour_id in tqdm.tqdm(self.tours, total=len(self), desc="Filter questions"):
            answers = []
            team_ids = []

            total_teams += len(self.tours[tour_id].teams)
            teams_with_mask = 0

            for team_id in self.tours[tour_id].teams:
                team = self.tours[tour_id][team_id]
                if team.mask is not None:
                    answers.append(team.mask)
                    team_ids.append(team_id)
                    teams_with_mask += 1

            if answers:
                total_questions_in_tour = max(map(len, answers))
                total_questions += total_questions_in_tour

                delete_row_indices = [row_num for row_num, answer_mask in enumerate(
                    answers) if total_questions_in_tour != len(answer_mask)]

                total_deleted_teams += len(delete_row_indices)

                for del_row_num in delete_row_indices:
                    team_id = team_ids[del_row_num]
                    self.tours[tour_id].teams.pop(team_id)
                    teams_with_mask -= 1

                answers = [answer for pos, answer in enumerate(answers) if pos not in delete_row_indices]
                team_ids = [team_id for pos, team_id in enumerate(team_ids) if pos not in delete_row_indices]

                # transpose table
                answers = tuple(zip(*answers))

                delete_row_indices = [row_index for row_index, answer_col in enumerate(
                    answers) if any(map(lambda x: x is None, answer_col))]

                total_none_questions += len(delete_row_indices)
                # transpose again return original order
                filtered_answer = tuple(
                    zip(*tuple(row for i, row in enumerate(answers) if i not in delete_row_indices)))

                assert len(filtered_answer) == teams_with_mask or len(filtered_answer) == 0

                for row_num, team_id in enumerate(team_ids):
                    assert self.tours[tour_id][team_id].mask is not None
                    self.tours[tour_id][team_id].mask = filtered_answer[row_num]

        LOGGER.info("Detect total incorrect answers %d from %d. Delete teams: %d of %d",
                    total_none_questions, total_questions, total_deleted_teams, total_teams)

    def to_player_dataframe(self, filter_by_mask: bool = False) -> pd.DataFrame:
        records = []
        global_answer_id_shift = 0

        data = None

        for tour_id in tqdm.tqdm(self.tours, total=len(self), desc="Convert to dataframe"):
            row = {"tour_id": tour_id}
            answer_shift = 0

            for team_id in self[tour_id]:
                team = self[tour_id][team_id]
                if (filter_by_mask and not team.mask) or not team.members:
                    continue

                if answer_shift == 0:
                    answer_shift = len(team.mask)

                row["team_id"] = team_id
                for player in team.members:
                    row["player_id"] = player.player_id
                    for local_answer_id, answer in enumerate(team.mask):
                        row["answer_id"] = global_answer_id_shift + local_answer_id
                        row["is_right_answer"] = answer
                        records.append(row.copy())

            if len(records) > 5000:
                if data is None:
                    data = pd.DataFrame.from_records(records)
                else:
                    data = data.append(pd.DataFrame.from_records(records), ignore_index=True)
                records.clear()

            global_answer_id_shift += answer_shift

        if records:
            data = data.append(pd.DataFrame.from_records(records))

        return data

    def to_team_rating_by_tour(self) -> pd.DataFrame:
        tours_res = []
        for tour_id in self:
            for team_id in self[tour_id].teams:
                team = self[tour_id][team_id]
                if team.members:
                    tours_res.append({"tour_id": tour_id, "members": tuple(
                        map(attrgetter("player_id"), team.members)), "team_id": team_id, "tour_rating": team.position})

        return pd.DataFrame.from_records(tours_res)
