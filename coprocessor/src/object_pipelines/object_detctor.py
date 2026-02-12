from math import atan
from typing import List, Optional

import backends.detector_coreml as coreml
import backends.detector_cpu as cpu
import backends.detector_cuda as cuda
from config.config import CameraConfig
from object_types import Observations, Results


class object_detector:
    results: Optional[List[Results]]
    observations: Observations

    def __init__(
        self,
        backend: str = "cpu",
        device: int = 0,
    ) -> None:
        match backend:
            case "cuda":
                self.detector = cuda.DetectorCUDA(device)
                pass
            case "coreML":
                self.detector = coreml.DetectorCoreML()
                pass
            case "hailo":
                pass
            case "rockchip":
                pass
            case "cpu":
                self.detector = cpu.DetectorCPU()
                pass
            case _:
                self.detector = cpu.DetectorCPU()
                pass

    def run_infrence(self, frame) -> None:
        self.results = self.detector.run_inference(frame)

    def get_results(self) -> Optional[List[Results]]:
        self.results = self.detector.get_results()
        return self.results

    def get_yaw(self, index: int) -> Optional[Observations]:
        if self.results is None:
            return None

        try:
            result = self.results[index]
            x1, y1, x2, y2 = result.bbox
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            cx = CameraConfig.camera_resolution_width / 2
            cy = CameraConfig.camera_resolution_height / 2
            x -= cx
            y -= cy
            f_x = CameraConfig.camera_matrix[0, 0]
            f_y = CameraConfig.camera_matrix[1, 1]

            yaw = atan(x / f_x)
            pitch = atan(y / f_y)

            return Observations(yaw=yaw, pitch=pitch)

        except (IndexError, AttributeError, TypeError):
            return None
