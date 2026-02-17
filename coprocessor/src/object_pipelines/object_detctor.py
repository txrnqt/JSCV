from math import atan2
from typing import List, Optional

import cv2
from config.config import CameraConfig
from object_pipelines.backends import detector_coreml as coreml
from object_pipelines.backends import detector_cpu as cpu
from object_pipelines.backends import detector_cuda as cuda
from object_pipelines.object_types import Observations, Results


class ObjectDetector:
    """
    High-level object detection interface with pluggable backends.

    This class wraps multiple detector implementations (CPU, CUDA, CoreML)
    behind a unified API. It runs inference, stores detection results, and
    provides utilities for converting detections into camera-relative
    observations and drawing bounding boxes.
    """

    results: Optional[List[Results]]
    observations: Observations

    def __init__(self, backend: str = "cpu", device: int = 0) -> None:
        """
        Initialize the object detector with the selected backend.

        Args:
            backend (str): Backend type to use ("cpu", "cuda", or "coreML").
            device (int): Device index for GPU/CUDA backends.
        """
        self.results = None
        match backend:
            case "cuda":
                self.detector = cuda.DetectorCUDA(device)
            case "coreML":
                self.detector = coreml.DetectorCoreML()
            case "cpu":
                self.detector = cpu.DetectorCPU()
            case _:
                self.detector = cpu.DetectorCPU()

    def run_inference(self, frame) -> None:
        """
        Run object detection inference on a frame.

        Args:
            frame: Input image frame compatible with the selected backend.

        Side Effects:
            Updates the internal results cache.
        """
        self.results = self.detector.run_inference(frame)

    def get_results(self) -> Optional[List[Results]]:
        """
        Retrieve the most recent detection results.

        Returns:
            Optional[List[Results]]: List of detection results or None if
            inference has not been run or failed.
        """
        return self.results

    def get_observations(self, index: int) -> Optional[Observations]:
        """
        Compute yaw and pitch angles for a specific detection.

        Angles are calculated relative to the camera center using the
        intrinsic camera matrix from CameraConfig.

        Args:
            index (int): Index of the detection in the results list.

        Returns:
            Optional[Observations]: Yaw and pitch angles for the detection,
            or None if unavailable or invalid.
        """
        if not self.results or index >= len(self.results):
            return None
        try:
            result = self.results[index]
            x1, y1, x2, y2 = result.bbox

            cx = CameraConfig.camera_resolution_width / 2
            cy = CameraConfig.camera_resolution_height / 2
            x = (x1 + x2) / 2 - cx
            y = (y1 + y2) / 2 - cy

            f_x = CameraConfig.camera_matrix[0, 0]
            f_y = CameraConfig.camera_matrix[1, 1]

            if f_x == 0 or f_y == 0:
                raise ValueError(
                    "Focal length values in camera_matrix must be non-zero."
                )

            yaw = atan2(x, f_x)
            pitch = atan2(y, f_y)
            return Observations(yaw=yaw, pitch=pitch)
        except (AttributeError, TypeError, ValueError):
            return None

    def plot(self, frame: cv2.Mat) -> cv2.Mat:
        """
        Draw bounding boxes for current detections on a frame.

        Args:
            frame (cv2.Mat): Input image frame.

        Returns:
            cv2.Mat: Frame with detection boxes drawn.
        """
        if not self.results:
            return frame

        output = frame.copy()
        for result in self.results:
            try:
                x1, y1, x2, y2 = map(int, result.bbox)
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except (AttributeError, TypeError, ValueError):
                continue

        return output
