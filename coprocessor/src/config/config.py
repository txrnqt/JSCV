from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy

# Base directory of the project (two levels above this file)
_BASE = Path(__file__).parent.parent


@dataclass
class ServerConfig:
    """
    Configuration for the vision server networking and feature toggles.

    Attributes:
        device_id (str): Unique identifier for the device running the server.
        server_ip (str): IP address used for server communication.
        april_tag_port (int): Network port for AprilTag tracking data.
        obj_port (int): Network port for object detection data.
        april_tag_tracking (bool): Enables or disables AprilTag tracking.
        obj_tracking (bool): Enables or disables object detection tracking.
    """

    device_id: str = ""
    server_ip: str = ""
    april_tag_port: int = 0
    obj_port: int = 0
    april_tag_tracking: bool = False
    obj_tracking: bool = False


@dataclass
class ObjConfig:
    """
    Configuration for the object detection subsystem.

    Attributes:
        backend (str): Name of the inference backend (e.g., CPU, GPU).
        model (str): Filesystem path to the object detection model.
            Defaults to a model package in the project's models directory.
        max_fps (float): Maximum frame rate allowed for object detection.
    """

    backend: str = ""
    model: str = str(_BASE / "models" / "jetson_orin_nano.mlpackage")
    max_fps: float = 0


@dataclass
class AprilTagConfig:
    """
    Configuration for AprilTag detection and tracking.

    Attributes:
        max_fps (float): Maximum frame rate for AprilTag processing.
        fiducial_size (float): Physical size of the AprilTag marker.
        fiducial_layout (Optional[float]): Identifier or configuration value
            describing the tag layout. Can be None if unused.
    """

    max_fps: float = 0
    fiducial_size: float = 0
    fiducial_layout: Optional[float] = 0


@dataclass
class CameraConfig:
    """
    Camera calibration and runtime configuration.

    Attributes:
        has_config (bool): Indicates whether calibration data is available.
        camera_matrix (numpy.ndarray): Intrinsic camera calibration matrix.
        distortion_coefficients (numpy.ndarray): Lens distortion parameters.
        camera_id (str): Identifier or name of the camera device.
        camera_resolution_width (int): Frame width in pixels.
        camera_resolution_height (int): Frame height in pixels.
        camera_auto_exposure (int): Auto-exposure mode or flag.
        camera_exposure (int): Manual exposure value.
        camera_gain (float): Camera gain setting.
        camera_denoise (float): Denoising strength applied to frames.
    """

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
    """
    Configuration for recording and event logging.

    Attributes:
        is_recording (bool): Enables or disables logging.
        time_stamp (int): Timestamp associated with the logging session.
        logging_location (str): Filesystem path where logs are stored.
        event_name (str): Name of the recorded event.
        match_type (str): Type or category of the event.
        match_number (int): Numeric identifier for the event.
    """

    is_recording: bool = True
    time_stamp: int = 0
    logging_location: str = ""
    event_name: str = ""
    match_type: str = ""
    match_number: int = 0
