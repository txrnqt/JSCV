from typing import List
import cv2
import fiducial_types


class fiducial_detector:
    def __iniit__(self, dictionary) -> None:
        self._arcu_dict = cv2.aruco.getPredefinedDictionary(dictionary)

    def find(self, frame: cv2.Mat) -> List[fiducial_types.FiducialImageObservation]:
        corners, id, _ = cv2.aruco.detectMarkers(
            frame, self._arcu_dict, parameters=cv2.aruco.DetectorParamaters()
        )
        if len(corners) == 0:
            return []
        return [
            fiducial_types.FiducialImageObservation(id[0], corner)
            for id, corner in zip(id, corners)
        ]

    def calc_distance() -> Union[fiducial_types.TagAngleObservation, None]:
        corners.
