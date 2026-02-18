from typing import List, Optional

import cv2
import numpy
from config.config import AprilTagConfig, CameraConfig
from fiducial_pipelines import fiducial_utils
from fiducial_pipelines.fiducial_types import (
    FiducialObservation,
    PoseObservation,
)
from wpimath.geometry import Pose3d, Quaternion, Rotation3d, Transform3d, Translation3d


class FiducialCpu:
    def __init__(self, dictionary_id: int):
        self.fiducial_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.observation: Optional[List[FiducialObservation]] = None

    def detect_fiducial(self, frame: cv2.Mat) -> None:
        """Detect fiducial markers in a frame."""
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame,
            self.fiducial_dict,
            parameters=cv2.aruco.DetectorParameters(),
        )

        if ids is None or len(corners) == 0:
            self.observation = None
            return

        self.observation = [
            FiducialObservation(tag_id[0], corner)
            for tag_id, corner in zip(ids, corners)
        ]

    def get_results(self) -> List[FiducialObservation]:
        """Return latest observations."""
        return self.observation if self.observation is not None else []

    def tag_result(self) -> Optional[PoseObservation]:
        """Route to the appropriate solver based on observation count."""
        if (
            not CameraConfig.has_config
            or self.observation is None
            or AprilTagConfig.fiducial_layout is None
        ):
            return None

        tag_ids, tag_poses, object_points, image_points = self.build_tag_data()

        if len(tag_ids) == 0:
            return None
        if len(tag_ids) == 1:
            return self.solve_single_tag(tag_ids, tag_poses, image_points)

        return self.solve_multi_tag(tag_ids, object_points, image_points)

    def get_single_tag_result(self) -> Optional[PoseObservation]:
        """Get pose result from a single detected tag."""
        if (
            not CameraConfig.has_config
            or self.observation is None
            or AprilTagConfig.fiducial_layout is None
            or len(self.observation) != 1
        ):
            return None

        tag_ids, tag_poses, _, image_points = self.build_tag_data()
        return self.solve_single_tag(tag_ids, tag_poses, image_points)

    def get_multi_tag_result(self) -> Optional[PoseObservation]:
        """Get pose result from multiple detected tags."""
        if (
            not CameraConfig.has_config
            or self.observation is None
            or AprilTagConfig.fiducial_layout is None
            or len(self.observation) < 2
        ):
            return None

        tag_ids, _, object_points, image_points = self.build_tag_data()
        return self.solve_multi_tag(tag_ids, object_points, image_points)

    def build_tag_data(self):
        """Match observed tags with layout poses and build PnP inputs."""
        fid_size = AprilTagConfig.fiducial_size
        layout = AprilTagConfig.fiducial_layout

        object_points = []
        image_points = []
        tag_ids = []
        tag_poses = []

        for observation in self.observation or []:
            for tag_data in layout["tags"]:
                if tag_data["ID"] != observation.tag_id:
                    continue

                tag_pose = Pose3d(
                    Translation3d(
                        tag_data["pose"]["translation"]["x"],
                        tag_data["pose"]["translation"]["y"],
                        tag_data["pose"]["translation"]["z"],
                    ),
                    Rotation3d(
                        Quaternion(
                            tag_data["pose"]["rotation"]["quaternion"]["W"],
                            tag_data["pose"]["rotation"]["quaternion"]["X"],
                            tag_data["pose"]["rotation"]["quaternion"]["Y"],
                            tag_data["pose"]["rotation"]["quaternion"]["Z"],
                        )
                    ),
                )

                corner_offsets = [
                    (0, fid_size / 2.0, -fid_size / 2.0),
                    (0, -fid_size / 2.0, -fid_size / 2.0),
                    (0, -fid_size / 2.0, fid_size / 2.0),
                    (0, fid_size / 2.0, fid_size / 2.0),
                ]

                for i, offset in enumerate(corner_offsets):
                    corner_pose = tag_pose + Transform3d(
                        Translation3d(*offset), Rotation3d()
                    )
                    object_points.append(
                        fiducial_utils.wpilibTranslationToOpenCv(
                            corner_pose.translation()
                        )
                    )
                    image_points.append(
                        [
                            observation.corners[0][i][0],
                            observation.corners[0][i][1],
                        ]
                    )

                tag_ids.append(observation.tag_id)
                tag_poses.append(tag_pose)

        return tag_ids, tag_poses, object_points, image_points

    def solve_single_tag(
        self,
        tag_ids,
        tag_poses,
        image_points,
    ) -> Optional[PoseObservation]:
        """Solve pose for a single tag using IPPE_SQUARE."""
        fid_size = AprilTagConfig.fiducial_size

        object_points = numpy.array(
            [
                [-fid_size / 2.0, fid_size / 2.0, 0.0],
                [fid_size / 2.0, fid_size / 2.0, 0.0],
                [fid_size / 2.0, -fid_size / 2.0, 0.0],
                [-fid_size / 2.0, -fid_size / 2.0, 0.0],
            ]
        )

        try:
            _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                object_points,
                numpy.array(image_points),
                CameraConfig.camera_matrix,
                CameraConfig.distortion_coefficients,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except cv2.error:
            return None

        field_to_tag_pose = tag_poses[0]

        def build_camera_pose(tvec, rvec):
            cam_to_tag_pose = fiducial_utils.opencv_to_wpilib(tvec, rvec)
            cam_to_tag = Transform3d(
                cam_to_tag_pose.translation(),
                cam_to_tag_pose.rotation(),
            )
            field_to_camera = field_to_tag_pose.transformBy(cam_to_tag.inverse())
            return Pose3d(
                field_to_camera.translation(),
                field_to_camera.rotation(),
            )

        pose0 = build_camera_pose(tvecs[0], rvecs[0])
        pose1 = build_camera_pose(tvecs[1], rvecs[1])

        return PoseObservation(
            tag_ids,
            pose0,
            errors[0][0],
            pose1,
            errors[1][0],
        )

    def solve_multi_tag(
        self,
        tag_ids,
        object_points,
        image_points,
    ) -> Optional[PoseObservation]:
        """Solve pose using SQPNP across multiple tags."""
        try:
            _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                numpy.array(object_points),
                numpy.array(image_points),
                CameraConfig.camera_matrix,
                CameraConfig.distortion_coefficients,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error:
            return None

        cam_to_field_pose = fiducial_utils.opencv_to_wpilib(
            tvecs[0],
            rvecs[0],
        )

        cam_to_field = Transform3d(
            cam_to_field_pose.translation(),
            cam_to_field_pose.rotation(),
        )

        field_to_camera = cam_to_field.inverse()

        field_to_camera_pose = Pose3d(
            field_to_camera.translation(),
            field_to_camera.rotation(),
        )

        return PoseObservation(
            tag_ids,
            field_to_camera_pose,
            errors[0][0],
            None,
            None,
        )
