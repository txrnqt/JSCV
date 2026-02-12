from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Results:
    c_class: int
    confidance: float
    bbox: List[float]


# in radians
@dataclass(frozen=True)
class Observations:
    yaw: float
    pitch: float
