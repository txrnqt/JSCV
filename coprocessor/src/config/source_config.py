import json

import config
import cv2
import ntcore
import numpy


class LocalSourceConfig:
    """
    Loads configuration data from local files and applies it to in-memory
    configuration objects.

    This class is responsible for reading static configuration sources such as
    JSON server configs and camera calibration files and updating the provided
    config dataclass instances.
    """

    def __init__(self, server_config_path: str):
        """
        Initialize a local configuration source.

        Args:
            server_config_path (str): Path to the JSON file containing
                server configuration values.
        """
        self.server_config_path = server_config_path

    def update_server_config(self, config_store: config.ServerConfig) -> None:
        """
        Load server configuration from a JSON file and update the given store.

        Args:
            config_store (config.ServerConfig): Target configuration object
                to populate with values from disk.
        """
        with open(self.server_config_path, "r") as config_file:
            config_data = json.loads(config_file.read())
            config_store.device_id = config_data["device_id"]
            config_store.server_ip = config_data["server_ip"]
            config_store.april_tag_port = config_data["april_tag_port"]
            config_store.obj_port = config_data["obj_port"]

    def update_camera_constants(
        self, config_store: config.CameraConfig, path: str
    ) -> None:
        """
        Load camera calibration constants from an OpenCV file and update
        the given camera configuration.

        The calibration file is expected to contain `camera_matrix` and
        `distortion_coefficients` nodes.

        Args:
            config_store (config.CameraConfig): Target camera configuration
                object to update.
            path (str): Path to the OpenCV calibration file.
        """
        calibration_store = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        camera_matrix = calibration_store.getNode("camera_matrix").mat()
        distortion_coefficients = calibration_store.getNode(
            "distortion_coefficients"
        ).mat()
        calibration_store.release()

        if (
            type(camera_matrix) is numpy.ndarray
            and type(distortion_coefficients) is numpy.ndarray
        ):
            config_store.camera_matrix = camera_matrix
            config_store.distortion_coefficients = distortion_coefficients
            config_store.has_config = True


class NTConfigSource:
    """
    Retrieves configuration values from NetworkTables and synchronizes them
    with global configuration objects.

    This class lazily initializes NetworkTables subscribers and continuously
    updates configuration values from network-published topics.
    """

    _init_complete: bool = False
    _camera_id_sub: ntcore.StringSubscriber
    _camera_resolution_width_sub: ntcore.IntegerSubscriber
    _camera_resolution_height_sub: ntcore.IntegerSubscriber
    _camera_auto_exposure_sub: ntcore.IntegerSubscriber
    _camera_exposure_sub: ntcore.DoubleSubscriber
    _camera_gain_sub: ntcore.DoubleSubscriber
    _camera_denoise_sub: ntcore.DoubleSubscriber
    _is_recording_sub: ntcore.BooleanSubscriber
    _time_stamp_sub: ntcore.DoubleSubscriber
    _logging_location_sub: ntcore.StringSubscriber
    _event_name_sub: ntcore.StringSubscriber
    _match_type_sub: ntcore.StringSubscriber
    _match_number_sub: ntcore.IntegerSubscriber
    _fiducial_layout_sub: ntcore.DoubleSubscriber
    _fiducial_size_sub: ntcore.DoubleSubscriber
    _backend_type_obj: ntcore.StringSubscriber

    def update(self) -> None:
        """
        Initialize NetworkTables subscriptions (if needed) and update all
        configuration values from their corresponding topics.

        This method should be called periodically to keep configuration
        synchronized with NetworkTables.
        """
        if not self._init_complete:
            nt_table = ntcore.NetworkTableInsrance.getDefault().getTable(
                "/" + config.ServerConfig.device_id + "/config"
            )

            # Camera configuration subscriptions
            self._camera_id_sub = nt_table.getStringTopic("camera_id").subscribe(
                config.CameraConfig.camera_id
            )
            self._camera_resolution_width_sub = ntcore.getIntergerTopic(
                "camera_resolution_width"
            ).subscribe(config.CameraConfig.camera_resolution_width)
            self._camera_resolution_height_sub = ntcore.getIntergerTopic(
                "camera_resolution_height"
            ).subscribe(config.CameraConfig.camera_resolution_height)
            self._camera_auto_exposure_sub = ntcore.getIntergerTopic(
                "camera_auto_exposure"
            ).subscribe(config.CameraConfig.camera_auto_exposure)
            self._camera_exposure_sub = ntcore.getDoubleTopic(
                "camera_exposure"
            ).subscribe(config.CameraConfig.camera_exposure)
            self._camera_gain_sub = ntcore.getDoubleTopic("camera_gain").subscribe(
                config.CameraConfig.camera_gain
            )
            self._camera_denoise_sub = ntcore.getDoubleTopic(
                "camera_denoise"
            ).subscribe(config.CameraConfig.camera_denoise)

            # Logging configuration subscriptions
            self._is_recording_sub = ntcore.getBooleanTopic("is_recording").subscribe(
                config.LoggingConfig.is_recording
            )
            self._time_stamp_sub = ntcore.getDoubleTopic("time_stamp").subscribe(
                config.LoggingConfig.time_stamp
            )
            self._logging_location_sub = ntcore.getStringTopic(
                "logging_location"
            ).subscribe(config.LoggingConfig.logging_location)
            self._event_name_sub = ntcore.getStringTopic("event_name").subscribe(
                config.LoggingConfig.event_name
            )
            self._match_type_sub = ntcore.getStringTopic("match_type").subscribe(
                config.LoggingConfig.match_type
            )
            self._match_number_sub = ntcore.getIntergerTopic("match_number").subscribe(
                config.LoggingConfig.match_number
            )

            # AprilTag configuration subscriptions
            self._fiducial_layout_sub = ntcore.getDoubleTopic(
                "fiducial_layout"
            ).subscribe(config.AprilTagConfig.fiducial_layout)
            self._fiducial_size_sub = ntcore.getDoubleTopic("fiducial_size").subscribe(
                config.AprilTagConfig.fiducial_size
            )

            # Object detection configuration subscriptions
            self._backend_type_obj = ntcore.getStringTopic(
                "backend_type_object"
            ).subscribe(config.ObjConfig.backend)

            self._init_complete = True

        # Apply latest values to config stores
        config.CameraConfig.camera_id = self._camera_id_sub.get()
        config.CameraConfig.camera_resolution_width = (
            self._camera_resolution_width_sub.get()
        )
        config.CameraConfig.camera_resolution_height = (
            self._camera_resolution_height_sub.get()
        )
        config.CameraConfig.camera_auto_exposure = self._camera_auto_exposure_sub.get()
        config.CameraConfig.camera_exposure = self._camera_exposure_sub.get()
        config.CameraConfig.camera_gain = self._camera_gain_sub.get()
        config.CameraConfig.camera_denoise = self._camera_denoise_sub.get()

        config.LoggingConfig.is_recording = self._is_recording_sub.get()
        config.LoggingConfig.time_stamp = self._time_stamp_sub.get()
        config.LoggingConfig.logging_location = self._logging_location_sub.get()
        config.LoggingConfig.event_name = self._event_name_sub.get()
        config.LoggingConfig.match_type = self._match_type_sub.get()
        config.LoggingConfig.match_number = self._match_number_sub.get()

        config.AprilTagConfig.fiducial_size = self._fiducial_size_sub.get()
        config.ObjConfig.backend = self._backend_type_obj.get()

        try:
            config.AprilTagConfig.fiducial_layout = self._fiducial_layout_sub.get()
        except Exception:
            config.AprilTagConfig.fiducial_layout = None
