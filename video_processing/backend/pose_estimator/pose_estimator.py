from typing import Any, List, Tuple

import cv2
import numpy as np
from openvino.inference_engine import IECore

from .hpe_associative_embedding import HpeAssociativeEmbedding
from .utils import OutputTransform


class PoseEstimator:
    """
    Данный класс предназначен для обнаружения позы человека в кадре.
    В качестве нейронной сети используется higher-hrnet
    """
    default_skeleton = (
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
        (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    )

    colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))

    point_names = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_heel',
        'right_heel',
    ]

    def draw_poses(self, img: np.ndarray, poses: np.ndarray, point_score_threshold: float = 0.1,
                   skeleton=default_skeleton) -> np.ndarray:
        """
        Метод отрисовки поз на изображении.
        """
        img = self._output_transform.resize(img)
        if poses.size == 0:
            return img
        stick_width = 4

        img_limbs = np.copy(img)
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points = self._output_transform.scale(points)
            points_scores = pose[:, 2]
            # Draw joints.
            for i, (p, v) in enumerate(zip(points, points_scores)):
                if v > point_score_threshold:
                    cv2.circle(img, tuple(p), 1, PoseEstimator.colors[i], 2)
            # Draw limbs.
            for i, j in skeleton:
                if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=PoseEstimator.colors[j],
                             thickness=stick_width)
        cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
        return img

    def get_annotated_poses(self, poses: np.ndarray, point_score_threshold: float = 0.1) -> List[dict]:
        annotated_poses = []
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points = self._output_transform.scale(points)
            points_scores = pose[:, 2]
            pack = zip(PoseEstimator.point_names, points, points_scores)
            annotated_poses.append({name: point for name, point, scores in pack if scores > point_score_threshold})
        return annotated_poses

    def __init__(self, frame_shape: tuple, device: str = 'CPU'):
        _model_path = 'backend/pose_estimator/higher-hrnet-w32/FP32/higher-hrnet-w32-human-pose-estimation.xml'
        _inference_engine = IECore()
        _aspect_ratio = frame_shape[1] / frame_shape[0]
        self._output_transform = OutputTransform(frame_shape, None)
        self._model = HpeAssociativeEmbedding(_inference_engine, _model_path, target_size=None,
                                              aspect_ratio=_aspect_ratio,
                                              prob_threshold=0.1, delta=0.5, padding_mode='center')
        self._exec_net = _inference_engine.load_network(network=self._model.net, device_name=device)

    def process_image(self, frame: np.ndarray) -> Tuple[Any, Any]:
        inputs, preprocessing_meta = self._model.preprocess(frame)
        prediction = self._exec_net.infer(inputs=inputs)
        poses, scores = self._model.postprocess(prediction, preprocessing_meta)
        return poses, scores
