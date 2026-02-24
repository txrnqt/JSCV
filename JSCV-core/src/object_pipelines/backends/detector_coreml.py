import traceback
from typing import List, Optional

import coremltools
import cv2
import numpy
from config.config import CameraConfig, ObjConfig
from object_pipelines.object_types import Observations, Results
from PIL import Image


class DetectorCoreML:
    """
    CoreML-based object detector wrapper.

    This class loads a CoreML model and provides utilities for preprocessing
    OpenCV frames, running inference, and returning filtered detection results.
    Detected objects are stored internally and can be retrieved after inference.
    """

    model: str = "JSCV-core/src/models/jetson_orin nano.mlpackage/Data/com.apple.CoreML/model.mlmodel"
    results: Optional[List[Results]] = None

    def __init__(self):
        """
        Initialize the detector and load the CoreML model defined in ObjConfig.
        """
        self.model = coremltools.models.MLModel(ObjConfig.model)

    def _to_rgb(self, frame: cv2.Mat) -> cv2.Mat:
        """
        Convert an OpenCV frame to a 3-channel RGB image.

        Handles grayscale and BGRA inputs and normalizes them to RGB format.

        Args:
            frame (cv2.Mat): Input OpenCV image frame.

        Returns:
            cv2.Mat: RGB-converted image.
        """
        if len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _letterbox(self, frame: cv2.Mat, size: int = 640):
        """
        Resize and pad an image to a square canvas while preserving aspect ratio.

        The image is scaled to fit within a square of the given size and padded
        with zeros (black pixels).

        Args:
            frame (cv2.Mat): Input RGB image.
            size (int): Target square dimension.

        Returns:
            tuple:
                - numpy.ndarray: Letterboxed image canvas.
                - float: Scaling factor applied to the original image.
                - int: Top padding in pixels.
                - int: Left padding in pixels.
        """
        h, w = frame.shape[:2]
        scale = size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        new_w = min(new_w, size)
        new_h = min(new_h, size)

        pad_top = (size - new_h) // 2
        pad_left = (size - new_w) // 2

        canvas = numpy.zeros((size, size, 3), dtype=numpy.uint8)
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = cv2.resize(
            frame, (new_w, new_h)
        )
        return canvas, scale, pad_top, pad_left

    def run_inference(
        self, frame: cv2.Mat, conf: float = 0.5
    ) -> Optional[List[Results]]:
        """
        Run object detection inference on a frame.

        The frame is converted to RGB, letterboxed, passed through the CoreML
        model, and filtered using a confidence threshold and non-maximum
        suppression.

        Args:
            frame (cv2.Mat): Input OpenCV frame.
            conf (float): Minimum confidence threshold for detections.

        Returns:
            Optional[List[Results]]: List of detection results, an empty list
            if no detections are found, or None if inference fails.
        """
        try:
            frame_rgb = self._to_rgb(frame)
            canvas, scale, pad_top, pad_left = self._letterbox(frame_rgb)
            image_coreml = Image.fromarray(canvas)
            inference = self.model.predict({"image": image_coreml})

            output = inference["var_1223"][0].T  # (8400, 5)

            scores = output[:, 4]
            mask = scores > conf
            output = output[mask]

            if len(output) == 0:
                self.results = []
                return self.results

            boxes = []
            for det in output:
                cx_c, cy_c, bw_c, bh_c, _ = det

                cx_f = (cx_c - pad_left) / scale
                cy_f = (cy_c - pad_top) / scale
                bw_f = bw_c / scale
                bh_f = bh_c / scale

                x_min = cx_f - bw_f / 2
                y_min = cy_f - bh_f / 2
                w = bw_f
                h = bh_f
                boxes.append([x_min, y_min, w, h])

            confidences = output[:, 4].tolist()
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, 0.45)

            observations: List[Results] = []
            for i in indices:
                x, y, w, h = boxes[i]
                observations.append(
                    Results(
                        c_class=0,
                        confidence=float(confidences[i]),
                        bbox=[x, y, x + w, y + h],
                    )
                )

            output = inference["var_1223"][0].T
            best = output[output[:, 4].argmax()]
            print("Best detection raw values:", best)

            self.results = observations
            return self.results

        except Exception as e:
            """
            Handle inference errors and store a None result state.
            """
            print(f"Inference error: {e}\n{traceback.format_exc()}")
            self.results = None
            return None

    def get_results(self) -> Optional[List[Results]]:
        """
        Retrieve the most recent inference results.

        Returns:
            Optional[List[Results]]: Cached detection results or None if
            inference has not succeeded.
        """
        return self.results
