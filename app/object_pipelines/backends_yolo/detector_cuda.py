from typing import Optional

from config.config import ObjConfig
from object_types import Results
from ultralytics import YOLO


class DetectorCUDA:
    model: YOLO
    results: Optional[Results]

    def __init__(self, device: int = 0) -> None:
        self.model = YOLO(ObjConfig.model)
        self.device = device

    def run_inference(self, frame, conf: float = 0.5) -> None:
        try:
            self.results = self.model(frame, device="cpu", conf=conf)
        except Exception as e:
            print(f"Inference error: {e}")
            self.results = None

    def get_results(self) -> Optional[Results]:
        return self.results
