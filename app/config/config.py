from dataclasses import dataclass
from typing import Optional

import numpy


@dataclass
class ServerConfig:
    device_id: str = ""
    server_ip: str = ""
    april_tag_port: int = 0
    obj_port: int = 0
    april_tag_tracking: bool = False
    obj_tracking: bool = False


@dataclass
class ObjConfig:
    model: str = ""
    max_fps: float = 0


@dataclass
class AprilTagConfig:
    max_fps: float = 0
    fiducial_size: float = 0
    fiducial_layout: Optional[float] = 0


@dataclass
class CameraConfig:
    has_config: bool = False
    camera_matrix: numpy.typing.NDArray[numpy.float64] = None
    distortion_coefficients: numpy.typing.NDArray[numpy.float64] = None
    camera_id: str = ""
    camera_resolution_width: int = 0
    camera_resolution_height: int = 0
    camera_auto_exposure: int = 0
    camera_exposure: int = 0
    camera_gain: float = 0
    camera_denoise: float = 0


@dataclass
class LoggingConfig:
    is_recording: bool = True
    time_stamp: int = 0
    logging_location: str = ""
    event_name: str = ""
    match_type: str = ""
    match_number: int = 0
