from typing import List, Optional

from config.config import ObjConfig
from object_types import Results
from ultralytics import YOLO
from ultralytics.engine.results import Results as YOLOResults


class DetectorCUDA:
    model: YOLO
    results: Optional[List[Results]]

    def __init__(self, device: int = 0) -> None:
        self.model = YOLO(ObjConfig.model)
        self.device = device

    def run_inference(self, frame, conf: float = 0.5) -> Optional[List[Results]]:
        try:
            yolo_results: List[YOLOResults] = self.model(frame, device="cpu", conf=conf)

            parsed_results: List[Results] = []

            for result in yolo_results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls = int(box.cls.item())
                    confidence = float(box.conf.item())
                    bbox = box.xyxy[0].tolist()

                    parsed_results.append(
                        Results(
                            c_class=cls,
                            confidance=confidence,
                            bbox=bbox,
                        )
                    )
            self.results = parsed_results
            return parsed_results

        except Exception as e:
            print(f"Inference error: {e}")
            self.results = None
            return None

    def get_results(self) -> Optional[List[Results]]:
        return self.results
