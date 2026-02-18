from typing import List

import cv2
import fiducial_pipelines.backends.fiducial_cpu as cpu
import fiducial_pipelines.fiducial_types as fiducial_types
from typing_extensions import Optional


class fiducial_detector:
    def __init__(self, backend: str, dictionary_id: int = 0) -> None:
        """Initialize the fiducial detector with specified backend."""
        match backend:
            case "cpu":
                self.detector = cpu.FiducialCpu(dictionary_id)
            case _:
                self.detector = cpu.FiducialCpu(dictionary_id)
        self.result = None

    def get_multi_tag_result(self):
        """Get pose result from multiple detected tags."""
        return self.detector.get_multi_tag_result()

    def get_single_tag_result(self):
        """Get pose result from a single detected tag."""
        return self.detector.get_single_tag_result()

    def find_fiducial(self, frame) -> None:
        """Detect fiducial markers in the frame."""
        self.detector.detect_fiducial(frame)

    def get_result(self) -> List[fiducial_types.FiducialObservation]:
        """Get the latest fiducial observations."""
        self.result = self.detector.get_results()
        return self.result

    def plot_result(
        self, frame, result: Optional[List[fiducial_types.FiducialObservation]]
    ) -> cv2.Mat:
        """Draw bounding boxes around detected fiducial markers and return the frame."""
        if result is None:
            return frame

        for r in result:
            corners = r.corners[0] if len(r.corners[0]) > 0 else r.corners
            (ptA, ptB, ptC, ptD) = corners

            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))

            cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
            cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
            cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
            cv2.line(frame, ptD, ptA, (0, 255, 0), 2)

            cv2.putText(
                frame, str(r.tag_id), ptA, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        return frame
