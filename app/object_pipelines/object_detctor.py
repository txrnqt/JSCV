from typing import Optional

import backends_yolo.detector_cpu as cpu
import backends_yolo.detector_cuda as cuda
from object_types import Results


class object_detector:
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

    def get_results(self) -> Optional[Results]:
        return self.detector.get_results()
