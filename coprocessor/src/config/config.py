from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy
import numpy.typing

_BASE = Path(__file__).parent.parent


@dataclass
class ServerConfig:
    """Vision server networking and feature configuration."""

    device_id: str = ""
    server_ip: str = ""
    april_tag_port: int = 0
    obj_port: int = 0
    april_tag_tracking: bool = False
    obj_tracking: bool = False


@dataclass
class ObjConfig:
    """Object detection subsystem configuration."""

    backend: str = ""
    model: str = str(_BASE / "models" / "jetson_orin_nano.mlpackage")
    max_fps: float = 0.0


@dataclass
class AprilTagConfig:
    """AprilTag detection and tracking configuration."""

    max_fps: float = 0.0
    fiducial_size: float = 0.0
    fiducial_layout: Optional[Any] = None


@dataclass
class CameraConfig:
    """Camera calibration and runtime configuration."""

    has_config: bool = False
    camera_matrix: Optional[numpy.typing.NDArray[numpy.float64]] = None
    distortion_coefficients: Optional[numpy.typing.NDArray[numpy.float64]] = None
    camera_id: str = ""
    camera_resolution_width: int = 0
    camera_resolution_height: int = 0
    camera_auto_exposure: int = 0
    camera_exposure: int = 0
    camera_gain: float = 0.0
    camera_denoise: float = 0.0


@dataclass
class LoggingConfig:
    """Recording and event logging configuration."""

    is_recording: bool = True
    time_stamp: int = 0
    logging_location: str = ""
    event_name: str = ""
    match_type: str = ""
    match_number: int = 0
