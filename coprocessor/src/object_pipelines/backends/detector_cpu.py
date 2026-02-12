from typing import List, Optional

from object_types import Results  # your dataclass
from ultralytics import YOLO
from ultralytics.engine.results import Results as YOLOResults


class DetectorCPU:
    model: YOLO

    def __init__(self, path: str = "app/models/yolo26n.pt") -> None:
        self.model = YOLO(path)

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
