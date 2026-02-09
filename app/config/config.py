from dataclasses import dataclass


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

@dataclass
class CameraConfig:

