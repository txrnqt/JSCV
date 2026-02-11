from math import atan
from typing import List, Optional

import backends_yolo.detector_cpu as cpu
import backends_yolo.detector_cuda as cuda
from config.config import CameraConfig
from object_types import Observations, Results


class object_detector:
    results: Results
    observations: Observations

    def __init__(
        self,
        backend: str = "cpu",
        device: int = 0,
        model: str = "app/models/yolo26n.pt",
    ) -> None:
        match backend:
            case "cuda":
                self.detector = cuda.DetectorCUDA(model, device)
                pass
            case "coreML":
                pass
            case "hailo":
                pass
            case "rockchip":
                pass
            case "cpu":
                self.detector = cpu.DetectorCPU(model)
                pass
            case _:
                self.detector = cpu.DetectorCPU()
                pass

    def run_infrence(self, frame) -> None:
        self.detector.run_inference(frame)
        self.results = self.detector.get_results()

    def get_results(self) -> Optional[Results]:
        self.results = self.detector.get_results()
        return self.results

    def get_yaw(self) -> Optional[Observations]:
        try:
            x1, y1, x2, y2 = self.results.bbox
            x = (x1 + x2) / 2
            x -= CameraConfig.camera_resolution_width
            y = (y1 + y2) / 2
            y -= CameraConfig.camera_resolution_height
            f_x = CameraConfig.camera_matrix[0, 0]

            yaw = atan(x / f_x)
            pitch = atan(x / f_x)

            self.observations = Observations(yaw, pitch)

            return self.observations
        except:
            return None
