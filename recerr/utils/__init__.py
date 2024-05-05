import yaml
from dataclasses import dataclass, field


DEFAULT_COLUMNS: list[str] = [
    "user_id",
    "item_id",
    "is_interaction",
    "is_recommendation",
]


@dataclass
class CriteriaGroup:
    name: str
    weight: float = 1.0
    criteria: list[str] = field(default_factory=list)


CriteriaLevel = list[CriteriaGroup]


def load_hierarchy(filepath: str) -> list[CriteriaLevel]:
    with open(filepath, mode="r") as file:
        config = read_hierarchy(file.read())

    return config


def read_hierarchy(config: str) -> list[CriteriaLevel]:
    return [[CriteriaGroup(**group) for group in level] for level in yaml.safe_load(config)]
