from typing import Optional

from ultralytics import YOLO


class DetectorCUDA:
    from typing import Optional

    def __init__(self, path: str = "app/models/yolo26n.pt", device: int = 0) -> None:
        self.model = YOLO(path)
        self.device = device

    def run_inference(self, frame, conf: float = 0.5):
        try:
            return self.model(frame, device=self.device, conf=conf)
        except Exception as e:
            print(f"Inference error: {e}")
            return None
