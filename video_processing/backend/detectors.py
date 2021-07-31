from typing import List

from .person import Person


class SimpleRaisedArmsDetector:
    """Класс обнаружения поднятых рук на изображении."""
    _keypoint_names = ['right_wrist', 'right_elbow', 'right_shoulder', 'left_wrist', 'left_elbow', 'left_shoulder']

    def detect(self, annotated_poses: List[dict]) -> List[bool]:
        """При проверке учитывыется, что начало координат - левый верхний угол."""
        detected_poses = []
        for pose in annotated_poses:
            if set(self._keypoint_names).issubset(pose.keys()):
                right_wrist_y = pose['right_wrist'][1]
                right_elbow_y = pose['right_elbow'][1]
                right_shoulder_y = pose['right_shoulder'][1]
                left_wrist_y = pose['left_wrist'][1]
                left_elbow_y = pose['left_elbow'][1]
                left_shoulder_y = pose['left_shoulder'][1]

                r_arm_raised = right_wrist_y < right_elbow_y < right_shoulder_y
                l_arm_raised = left_wrist_y < left_elbow_y < left_shoulder_y
                detected_poses.append(r_arm_raised and l_arm_raised)
            else:
                detected_poses.append(False)
        return detected_poses


class RaisingArmsMomentDetector:
    """Класс отслеживания момента поднятия рук."""

    # Число кадров, в течении которого флаг поднятых рук не сбрасывается в отсутсвии обнаружения.
    _no_detection_count = 5

    def detect(self, skeletons_data: List[Person], poses_detected: List[bool]) -> List[bool]:
        unique_poses = []
        for skeleton_data, pose_detected in zip(skeletons_data, poses_detected):
            if pose_detected:
                if skeleton_data.unique_pose:
                    skeleton_data.unique_pose = False
                    unique_poses.append(True)
                else:
                    unique_poses.append(False)
            else:
                if not skeleton_data.no_detection_count:
                    skeleton_data.no_detection_count = self._no_detection_count
                skeleton_data.no_detection_count -= 1
                if skeleton_data.no_detection_count == 0:
                    skeleton_data.no_detection_count = self._no_detection_count
                    skeleton_data.unique_pose = True
                unique_poses.append(False)
        return unique_poses
