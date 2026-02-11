from typing import Optional
from ultralytics import YOLO
from object_types import Results

class DetectorCUDA:
    model: YOLO
    results: Optional[Results]

    def __init__(self, path: str = "app/models/yolo26n.pt", device: int = 0) -> None:
        self.model = YOLO(path)
        self.device = device


    def run_inference(self, frame, conf: float = 0.5) -> None:
            try:
                self.results = self.model(frame, device="cpu", conf=conf)
            except Exception as e:
                print(f"Inference error: {e}")
                self.results = None

        def get_results(self) -> Optional[Results]:
            return self.results
