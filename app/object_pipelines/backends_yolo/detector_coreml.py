from typing import List, Optional, Union

import coremltools
import cv2
import numpy
from config.config import CameraConfig, ObjConfig
from object_types import Observations, Results
from PIL import Image


class DetectorCoreML:
    model: Optional[coremltools.models.MLModel] = None
    results: Optional[List[Results]] = None

    def __init__(self):
        self.model = coremltools.models.MLModel(ObjConfig.model)

    def run_inference(self, frame: cv2.Mat, conf: float = 0.5) -> None:
        try:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            image_scaled = numpy.zeros((640, 640, 3), dtype=numpy.uint8)

            scaled_height = int(640 / (frame.shape[1] / frame.shape[0]))
            bar_height = int((640 - scaled_height) / 2)

            image_scaled[bar_height : bar_height + scaled_height, 0:640] = cv2.resize(
                frame, (640, scaled_height)
            )

            image_coreml = Image.fromarray(image_scaled)

            inference = self.model.predict({"image": image_coreml})

            observations: List[Results] = []

            for coordinates, confidence in zip(
                inference["coordinates"], inference["confidence"]
            ):
                obj_class = max(range(len(confidence)), key=confidence.__getitem__)
                confidence = float(confidence[obj_class])

                if confidence < conf:
                    continue

                x = coordinates[0] * frame.shape[1]
                y = ((coordinates[1] * 640 - bar_height) / scaled_height) * frame.shape[
                    0
                ]
                width = coordinates[2] * frame.shape[1]
                height = coordinates[3] / (scaled_height / 640) * frame.shape[0]

                x_min = x - width / 2
                y_min = y - height / 2
                x_max = x + width / 2
                y_max = y + height / 2

                bbox = [x_min, y_min, x_max, y_max]

                observations.append(
                    Results(c_class=obj_class, confidance=confidence, bbox=bbox)
                )

            self.results = observations

        except Exception as e:
            print(f"Inference error: {e}")
            self.results = None
