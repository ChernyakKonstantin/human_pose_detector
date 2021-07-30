class SimpleRaisedArmsDetector:
    """Класс обнаружения поднятых рук на изображении."""
    reference_keypoint_name = 'nose'
    keypoint_names = ['right_wrist', 'right_elbow', 'left_wrist', 'left_elbow', ]

    def detect(self, annotated_poses: list) -> list:
        detected_poses = []
        for pose in annotated_poses:
            try:
                if all([pose[self.reference_keypoint_name][1] > pose[kp_name][1] for kp_name in self.keypoint_names]):
                    detected_poses.append(pose)
            except KeyError:
                pass
        return detected_poses


class UniqueDetector:
    """Класс слежения, что руки были подняты и опущены."""
    def __init__(self):
        pass
