from typing import List


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
        return detected_poses


class RaisingArmsMomentDetector:
    """Класс отслеживания момента поднятия рук."""

    # Квадарт максимального сдвига ограничетельной рамки (чтобы не считать sqrt).
    _tracking_threshold = 50 ^ 2
    # Число кадров, в течении которого флаг поднятых рук не сбрасывается в отсутсвии обнаружения.
    _no_detection_count = 5

    def __init__(self):
        self._records = []

    def detect(self, bounding_boxes: List[dict], skeletons_detected: List[bool], ) -> List[int]:
        if len(self._records) == 0:
            item = {'unique': True, 'no_detection_counter': self._no_detection_count}
            self._records.extend([item for _ in skeletons_detected])
        idx = []
        for index, (bbox, skeleton_detected) in enumerate(zip(bounding_boxes, skeletons_detected)):
            if skeleton_detected:
                if self._records[index]['unique']:
                    self._records[index]['unique'] = False
                    idx.append(index)
            else:
                self._records[index]['no_detection_counter'] -= 1
                if self._records[index]['no_detection_counter'] == 0:
                    self._records[index]['no_detection_counter'] = self._no_detection_count
                    self._records[index]['unique'] = True
        return idx
