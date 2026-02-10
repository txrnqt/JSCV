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
    _camera_denoise_sub: ntcore.DoubleSubscriber
    _is_recording_sub: ntcore.BooleanSubscriber
    _time_stamp_sub: ntcore.DoubleSubscriber
    _logging_location: ntcore.StringSubscriber
    _event_name: ntcore.StringSubscriber
    _match_type: ntcore.StringSubscriber
    _match_number: ntcore.IntergerSubscriber
    _fiducial_layout: ntcore.DoubleSubscriber
    _fiducial_size: ntcore.DoubleSubscriber

    def update(self):
        if not self._init_complete:
            nt_table = ntcore.NetworkTableInsrance.getDefault().getTable("/" + config.ServerConfig.device_id + "/config")
            self._camera_id_sub = nt_table.getStringTopic("camera_id").subscribe(config.camera_config.camera_id)
            self._camera_resolution_width_sub = ntcore.getIntergerTopic("camera_resolution_width").subscribe(config.CameraConfig.camera_resolution_width)
            self._camera_resolution_height_sub = ntcore.getIntergerTopic("camera_resolution_height").subscribe(config.CameraConfig.camera_resolution_height)
            self._camera_auto_exposure_sub = ntcore.getIntergerTopic("camera_auto_exposure").subscribe(config.CameraConfig.camera_auto_exposure)
            self._camera_exposure_sub = ntcore.getDoubleTopic("camera_exposure").subscribe(config.CameraConfig.camera_exposure)
            self._camera_gain_sub = ntcore.getDoubleTopic("camera_gain").subscribe(config.CameraConfig.camera_gain)
            self._camera_denoise_sub = ntcore.getDoubleTopic("camera_denoise").subscribe(config.CameraConfig.camera_denoise)
            self._is_recording_sub = ntcore.getBooleanTopic("is_recording").subscribe(config.LoggingConfig.is_recording)
            self._time_stamp_sub = ntcore.getDoubleTopic("time_stamp").subscribe(config.LoggingConfig.time_stamp)
            self._logging_location = ntcore.getStringTopic("logging_location").subscribe(config.LoggingConfig.logging_location)
            self._event_name = ntcore.getStringTopic("event_name").subscribe(config.LoggingConfig.event_name)
            self._match_type = ntcore.getStringTopic("match_type").subscribe(config.LoggingConfig.match_type)
            self._match_number = ntcore.getIntergerTopic("match_number").subscribe(config.LoggingConfig.match_number)
            self._fiducial_layout = ntcore.getDoubleTopic("fiducial_layout").subscribe(config.AprilTagConfig.fiducial_layout)
            self._fiducial_size = ntcore.getDoubleTopic("fiducial_size").subscribe(config.AprilTagConfig.fiducial_size)
            self._init_complete = True

        config.cameraconfig.camera_id = self._camera_id_sub.get()
        config.cameraconfig.camera_resolution_width = self._camera_resolution_width_sub.get()
        config.cameraconfig.camera_resolution_height = self._camera_resolution_width_sub.get()
        config.cameraconfig.camera_auto_exposure = self._camera_auto_exposure_sub.get()
        config.cameraconfig.camera_id = self._camera_id_sub.get()
        config.cameraconfig.camera_id = self._camera_id_sub.get()
        config.cameraconfig.camera_id = self._camera_id_sub.get()
        config.cameraconfig.camera_id = self._camera_id_sub.get()
        config.cameraconfig.camera_id = self._camera_id_sub.get()
        config.cameraconfig.camera_id = self._camera_id_sub.get()
        config.cameraconfig.camera_id = self._camera_id_sub.get()
        config.cameraconfig.camera_id = self._camera_id_sub.get()
        config.cameraconfig.camera_id = self._camera_id_sub.get()
        config.cameraconfig.camera_id = self._camera_id_sub.get()
        config.cameraconfig.camera_id = self._camera_id_sub.get()
         

        
