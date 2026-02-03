from ultralytics import YOLO


class DetectorCPU:
    from typing import Optional

    def __init__(self, path: str = "app/models/yolo26n.pt") -> None:
        self.model = YOLO(path)

    def run_inference(self, frame, conf: float = 0.5):
        try:
            return self.model(frame, device="cpu", conf=conf)
        except Exception as e:
            print(f"Inference error: {e}")
            return None
