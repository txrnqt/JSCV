from dataclasses import dataclass
from typing import List


@dataclass
class Results:
    """
    Represents a single object detection result.

    Attributes:
        c_class (int): Predicted class index for the detected object.
        confidence (float): Confidence score of the detection.
        bbox (list): Bounding box coordinates in the format
            [x1, y1, x2, y2].
    """

    c_class: int
    confidence: float
    bbox: list


# in radians
@dataclass(frozen=True)
class Observations:
    """
    Immutable angular observation of a detected object.

    The angles represent the object's position relative to the
    camera center.

    Attributes:
        yaw (float): Horizontal angle in radians.
        pitch (float): Vertical angle in radians.
    """

    yaw: float
    pitch: float
