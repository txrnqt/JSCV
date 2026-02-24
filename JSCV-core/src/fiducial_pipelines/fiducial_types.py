from dataclasses import dataclass
from typing import List, Union

import numpy
from wpimath.geometry import Pose3d


@dataclass(frozen=True)
class FiducialObservation:
    tag_id: int
    corners: numpy.typing.NDArray[numpy.float64]


@dataclass(frozen=True)
class TagAngleObservation:
    tag_id: int
    corners: numpy.typing.NDArray[numpy.float64]
    distance: float


@dataclass(frozen=True)
class PoseObservation:
    tag_ids: List[int]
    pose_0: Pose3d
    error_0: float
    pose_1: Union[Pose3d, None]
    error_1: Union[float, None]
