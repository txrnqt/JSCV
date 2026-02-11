from typing import Optional

from object_types import Results
from ultralytics import YOLO


class DetectorCPU:
    model: YOLO
    results: Optional[Results]

    def __init__(self, path: str = "app/models/yolo26n.pt") -> None:
        self.model = YOLO(path)

    def run_inference(self, frame, conf: float = 0.5) -> None:
        try:
            self.results = self.model(frame, device="cpu", conf=conf)
        except Exception as e:
            print(f"Inference error: {e}")
            self.results = None

    def get_results(self) -> Optional[Results]:
        return self.results
