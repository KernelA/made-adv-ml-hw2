import pickle
import warnings
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional

DATA_KEY_NAME = "key_path"


def init_from_data(cls, data: dict):
    kv_pairs = {}

    for field in fields(cls):
        if DATA_KEY_NAME in field.metadata:
            key_path = list(field.metadata[DATA_KEY_NAME])
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
    player_id: int = field(metadata={DATA_KEY_NAME: ("player", "id")})
    rating: int = field(metadata={DATA_KEY_NAME: ("rating",)})
    used_rating: int = field(metadata={DATA_KEY_NAME: ("usedRating", )})

    @ classmethod
    def from_dict(cls, data: dict) -> "Player":
        return init_from_data(cls, data)


@ dataclass
class Team:
    team_id: int = field(metadata={DATA_KEY_NAME: ("team", "id")})
    name: str = field(metadata={DATA_KEY_NAME: ("team", "name")})
    mask: Optional[List[bool]] = field(metadata={DATA_KEY_NAME: ("mask",)})
    position: int = field(metadata={DATA_KEY_NAME: ("position",)})
    members: List[Player] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.mask is not None:
            self.mask = list(map(bool, self.mask))

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

    @classmethod
    def from_dict(cls, data: List[dict]) -> "Teams":
        teams = cls()

        for team_data in data:
            team = Team.from_dict(team_data)
            if team.mask is not None and team.members:
                teams.add_team(team)

        return teams

    def __len__(self):
        return len(self.teams)

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
