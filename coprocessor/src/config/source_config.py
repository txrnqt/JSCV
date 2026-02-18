import json
import config
import cv2
import ntcore
import numpy


class LocalSourceConfig:
    """
    Loads configuration data from local files and applies it to in-memory
    configuration objects.
    """

    def __init__(self, server_config_path: str):
        """Initialize a local configuration source."""
        self.server_config_path = server_config_path

    def update_server_config(self, config_store: config.ServerConfig) -> None:
        """Load server configuration from JSON into the provided store."""
        with open(self.server_config_path, "r", encoding="utf-8") as config_file:
            config_data = json.load(config_file)

        config_store.device_id = config_data.get("device_id", "")
        config_store.server_ip = config_data.get("server_ip", "")
        config_store.april_tag_port = config_data.get("april_tag_port", 0)
        config_store.obj_port = config_data.get("obj_port", 0)
        config_store.april_tag_tracking = config_data.get("april_tag_tracking", False)
        config_store.obj_tracking = config_data.get("obj_tracking", False)

    def update_camera_constants(
        self, config_store: config.CameraConfig, path: str
    ) -> None:
        """Load OpenCV camera calibration data into the provided store."""
        calibration_store = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        camera_matrix = calibration_store.getNode("camera_matrix").mat()
        distortion_coefficients = calibration_store.getNode(
            "distortion_coefficients"
        ).mat()

        calibration_store.release()

        if isinstance(camera_matrix, numpy.ndarray) and isinstance(
            distortion_coefficients, numpy.ndarray
        ):
            config_store.camera_matrix = camera_matrix
            config_store.distortion_coefficients = distortion_coefficients
            config_store.has_config = True


class NTConfigSource:
    """
    Synchronizes configuration values from NetworkTables into config objects.
    """

    def __init__(
        self,
        server_config: config.ServerConfig,
        camera_config: config.CameraConfig,
        logging_config: config.LoggingConfig,
        apriltag_config: config.AprilTagConfig,
        obj_config: config.ObjConfig,
    ):
        self._server_config = server_config
        self._camera_config = camera_config
        self._logging_config = logging_config
        self._apriltag_config = apriltag_config
        self._obj_config = obj_config

        self._init_complete = False

    def update(self) -> None:
        """Initialize subscriptions if needed and pull latest values."""
        if not self._init_complete:
            nt_instance = ntcore.NetworkTableInstance.getDefault()
            nt_table = nt_instance.getTable(
                "/" + self._server_config.device_id + "/config"
            )

            # Camera subscriptions
            self._camera_id_sub = nt_table.getStringTopic("camera_id").subscribe(
                self._camera_config.camera_id
            )
            self._camera_resolution_width_sub = nt_table.getIntegerTopic(
                "camera_resolution_width"
            ).subscribe(self._camera_config.camera_resolution_width)
            self._camera_resolution_height_sub = nt_table.getIntegerTopic(
                "camera_resolution_height"
            ).subscribe(self._camera_config.camera_resolution_height)
            self._camera_auto_exposure_sub = nt_table.getIntegerTopic(
                "camera_auto_exposure"
            ).subscribe(self._camera_config.camera_auto_exposure)
            self._camera_exposure_sub = nt_table.getDoubleTopic(
                "camera_exposure"
            ).subscribe(self._camera_config.camera_exposure)
            self._camera_gain_sub = nt_table.getDoubleTopic("camera_gain").subscribe(
                self._camera_config.camera_gain
            )
            self._camera_denoise_sub = nt_table.getDoubleTopic(
                "camera_denoise"
            ).subscribe(self._camera_config.camera_denoise)

            # Logging subscriptions
            self._is_recording_sub = nt_table.getBooleanTopic("is_recording").subscribe(
                self._logging_config.is_recording
            )
            self._time_stamp_sub = nt_table.getDoubleTopic("time_stamp").subscribe(
                self._logging_config.time_stamp
            )
            self._logging_location_sub = nt_table.getStringTopic(
                "logging_location"
            ).subscribe(self._logging_config.logging_location)
            self._event_name_sub = nt_table.getStringTopic("event_name").subscribe(
                self._logging_config.event_name
            )
            self._match_type_sub = nt_table.getStringTopic("match_type").subscribe(
                self._logging_config.match_type
            )
            self._match_number_sub = nt_table.getIntegerTopic("match_number").subscribe(
                self._logging_config.match_number
            )

            # AprilTag subscriptions
            self._fiducial_layout_sub = nt_table.getDoubleTopic(
                "fiducial_layout"
            ).subscribe(0.0)
            self._fiducial_size_sub = nt_table.getDoubleTopic(
                "fiducial_size"
            ).subscribe(self._apriltag_config.fiducial_size)

            # Object detection subscriptions
            self._backend_type_obj = nt_table.getStringTopic(
                "backend_type_object"
            ).subscribe(self._obj_config.backend)

            self._init_complete = True

        # Apply values
        self._camera_config.camera_id = self._camera_id_sub.get()
        self._camera_config.camera_resolution_width = (
            self._camera_resolution_width_sub.get()
        )
        self._camera_config.camera_resolution_height = (
            self._camera_resolution_height_sub.get()
        )
        self._camera_config.camera_auto_exposure = self._camera_auto_exposure_sub.get()
        self._camera_config.camera_exposure = self._camera_exposure_sub.get()
        self._camera_config.camera_gain = self._camera_gain_sub.get()
        self._camera_config.camera_denoise = self._camera_denoise_sub.get()

        self._logging_config.is_recording = self._is_recording_sub.get()
        self._logging_config.time_stamp = int(self._time_stamp_sub.get())
        self._logging_config.logging_location = self._logging_location_sub.get()
        self._logging_config.event_name = self._event_name_sub.get()
        self._logging_config.match_type = self._match_type_sub.get()
        self._logging_config.match_number = self._match_number_sub.get()

        self._apriltag_config.fiducial_size = self._fiducial_size_sub.get()
        self._obj_config.backend = self._backend_type_obj.get()

        try:
            self._apriltag_config.fiducial_layout = self._fiducial_layout_sub.get()
        except Exception:
            self._apriltag_config.fiducial_layout = None


def update_camera_constants_test(
    config_store: config.CameraConfig,
) -> None:
    """Load test calibration data into a camera config."""
    calibration_store = cv2.FileStorage(
        "coprocessor/src/test/camera_calibration.json",
        cv2.FILE_STORAGE_READ,
    )

    camera_matrix = calibration_store.getNode("camera_matrix").mat()
    distortion_coefficients = calibration_store.getNode("distortion_coefficients").mat()

    calibration_store.release()

    if isinstance(camera_matrix, numpy.ndarray) and isinstance(
        distortion_coefficients, numpy.ndarray
    ):
        config_store.camera_matrix = camera_matrix
        config_store.distortion_coefficients = distortion_coefficients
        config_store.has_config = True
