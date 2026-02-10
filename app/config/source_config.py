import json
import cv2
import numpy
import config


class LocalSourceConfig:
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
        camera_matrix = calibration_store.getNode("camera_matrix").mat()
        distortion_coefficients = calibration_store.getNode("distortion_coefficients").mat()
        calibration_store.release()
        
        if type(camera_matrix) == numpy.ndarray and type(distortion_coefficients) == numpy.ndarray:
            config_store.camera_matrix = camera_matrix
            config_store.distortion_coefficients = distortion_coefficients
            config_store.has_config = True

class NTConfigSource():
    _init_complete: bool = False
    _camera_id_sub: ntcore.StringSubscriber
    _camera_resolution_width_sub: ntcore.IntergerSubscriber
    _camera_resolution_height_sub: ntcore.IntergerSubscriber
    _camera_auto_exposure_sub: ntcore.IntergerSubscriber
    _camera_exposure_sub: ntcore.DoubleSubscriber
    _camera_gain_sub: ntcore.DoubleSubscriber
