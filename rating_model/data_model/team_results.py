import pickle
import warnings
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple, MutableSet

import pandas as pd

KEY_PATH = "key_path"


def init_from_data(cls, data: dict):
    kv_pairs = {}

    for field in fields(cls):
        if KEY_PATH in field.metadata:
            key_path = list(field.metadata[KEY_PATH])
            try:
                data_value = data[key_path.pop(0)]

                while key_path:
                    data_value = data_value[key_path.pop(0)]
            except KeyError as exc:
                data_value = None
                warnings.warn(
                    f"Detect None value in path {str(exc)}.")

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
    mask: Optional[Tuple[bool]] = field(metadata={KEY_PATH: ("mask",)})
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
                    new_values.append(False)
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
            if team.mask is not None and team.members:
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
    results: Dict[int, Teams] = field(init=False, default_factory=dict)

    @ staticmethod
    def load_pickle(file_obj) -> "TeamResults":
        data = pickle.load(file_obj)

        results = TeamResults()

        for tour_id in data:
            teams = Teams.from_dict(data[tour_id])
            if len(teams) > 0:
                results.add_result(tour_id, teams)

        return results

    def add_result(self, result_id: int, teams: Teams):
        self.results[result_id] = teams

    def __len__(self):
        return len(self.results)

    def __getitem__(self, result_id: int) -> Teams:
        return self.results[result_id]

    def to_player_dataframe(self, tours_ids: MutableSet[int] = None) -> pd.DataFrame:
        records = []
        global_answer_id_shift = 0

        data = None

        if tours_ids is not None:
            tours = filter(lambda x: x in tours_ids, self.results.keys())
        else:
            tours = self.results.keys()

        for tour_id in tours:
            row = {"tour_id": tour_id}
            answer_shift = None
            for team_id in self[tour_id]:
                if answer_shift is None:
                    answer_shift = len(self[tour_id][team_id].mask)

                row["team_id"] = team_id
                for player in self.results[tour_id][team_id].members:
                    row["player_id"] = player.player_id
                    for local_answer_id, answer in enumerate(self[tour_id][team_id].mask):
                        row["answer_id"] = global_answer_id_shift + local_answer_id
                        row["is_right_answer"] = answer
                        records.append(row.copy())

            if len(records) > 5000:
                if data is None:
                    data = pd.DataFrame.from_records(records)
                else:
                    data = data.append(pd.DataFrame.from_records(records))
                records.clear()

            global_answer_id_shift += answer_shift

        return data
