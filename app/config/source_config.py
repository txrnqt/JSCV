import json
import cv2
import numpy
import config


class SourceConfig:
    def __init__(self, server_config_path: str):
        self.server_config_path = server_config_path

    def update_server_config(self, config_store: config.ServerConfig) -> None:
        with open(self.server_config_path, "r") as config_file:
            config_data = json.loads(config_file.read())
            config_store.device_id = config_data["device_id"]
            config_store.server_ip = config_data["server_ip"]
            config_store.april_tag_port = config_data["april_tag_port"]
            config_store.obj_port = config_data["obj_port"]

    def update_camera_constants(
        self, config_store: config.CameraConfig, path: str
    ) -> None:
        calibration_store = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
