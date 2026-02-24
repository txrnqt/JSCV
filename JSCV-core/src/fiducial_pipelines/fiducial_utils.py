from wpilib.geometry import Pose3d, Translation3d, Rotation3d
import numpy
import math
from typing import List


def opencv_to_wpilib(
    tvecs: numpy.typing.NDArray[numpy.float64],
    rvec: numpy.typing.NDArray[numpy.float64],
) -> Pose3d:
    return Pose3d(
        Translation3d(tvecs[2][0], -tvecs[0][0], -tvecs[1][0]),
        Rotation3d(
            numpy.array([rvec[2][0], -rvec[0][0], -rvec[1][0]]),
            math.sqrt(
                math.pow(rvec[0][0], 2)
                + math.pow(rvec[1][0], 2)
                + math.pow(rvec[2][0], 2)
            ),
        ),
    )


def wpilib_to_opencv(translation: Translation3d) -> List[float]:
    return [-translation.Y(), -translation.Z(), translation.X()]
