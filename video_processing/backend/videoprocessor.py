import logging
from typing import List

import cv2 as cv
import numpy as np

from .database_handler.db_handler import DBHandler
from .detectors import RaisingArmsMomentDetector, SimpleRaisedArmsDetector
from .trackers import SkeletonTracker
from .pose_estimator import PoseEstimator
from .streamer import Streamer

logging.basicConfig(level=logging.DEBUG)


class VideoProcessor:
    """
    Класс обработчика видео.
    Функционал:
    1) Чтение видеофайла.
    2) Определение позы человека на видео.
    3) Определение, явялется ли поза искомой.
    4) Кадрирование человека в искомой позе.
    5) Отправка кадрированного изображения в БД.
    6) Стриминг изображения в отдельном потоке.
    """

    @staticmethod
    def _encode_image_to_jpg(img: np.ndarray) -> bytes:
        """ Метод преобразует numpy-массив изображения в файл заданного расширения и возвращает строку байтов."""
        ret, img_data = cv.imencode('.jpg', cv.cvtColor(img, cv.COLOR_RGB2BGR))
        return img_data.tobytes()

    @staticmethod
    def get_bounding_boxes(skeletons: List[dict]) -> List[dict]:
        bounding_boxes = []
        for skeleton in skeletons:
            x = [point[0] for point in skeleton.values()]
            y = [point[1] for point in skeleton.values()]
            bounding_box = {
                'min_x': min(x),
                'max_x': max(x),
                'min_y': min(y),
                'max_y': max(y),
            }
            bounding_boxes.append(bounding_box)
        return bounding_boxes

    @staticmethod
    def _resize(img: np.ndarray, max_dim_px: int = 100):
        factor = max_dim_px / max(img.shape)
        return cv.resize(img, dsize=(0, 0), fx=factor, fy=factor)

    @staticmethod
    def _crop_person(img: np.ndarray, bounding_box: dict) -> np.ndarray:
        min_x = bounding_box['min_x']
        max_x = bounding_box['max_x']
        min_y = bounding_box['min_y']
        max_y = bounding_box['max_y']
        return img[min_y: max_y, min_x: max_x]

    def __init__(self, input_video_file: str, db_name: str, db_user: str, db_password: str,
                 host: str = None, port: int = None, db_host: str = None, db_port: int = None):
        self._input_video_file = input_video_file
        self._video_reader = cv.VideoCapture(self._input_video_file)
        self._cur_frame = 0
        frame_shape = self._video_reader.read()[1].shape

        self._pose_estimator = PoseEstimator(frame_shape)
        self._pose_detector = SimpleRaisedArmsDetector()
        self._skeleton_tracker = SkeletonTracker()
        self._unique_detector = RaisingArmsMomentDetector()

        self._db_handler = DBHandler(db_name, db_user, db_password, db_host, db_port)
        self._streamer = Streamer(host, port, daemon=True)

        self._db_handler.connect()
        self._streamer.start()

    def _process_frame(self, img: np.ndarray) -> np.ndarray:
        poses, _ = self._pose_estimator.process_image(img)
        annotated_img = self._pose_estimator.draw_poses(img, poses)
        annotated_poses = self._pose_estimator.get_annotated_poses(poses)
        skeleton_idx = self._skeleton_tracker.track(annotated_poses)
        if skeleton_idx and max(skeleton_idx) > len(annotated_poses) - 1:
            return annotated_img
        annotated_poses = [annotated_poses[index] for index in skeleton_idx]  # Упорядочить скелеты
        skeletons_bounding_boxes = self.get_bounding_boxes(annotated_poses)
        skeletons_detected = self._pose_detector.detect(annotated_poses)
        unique_raising_arms_moment_skeletons_idx = self._unique_detector.detect(skeletons_bounding_boxes,
                                                                                skeletons_detected)
        for index in unique_raising_arms_moment_skeletons_idx:
            logging.debug(f'Raised arms were detected on skeleton: {index}!')
            bounding_box = skeletons_bounding_boxes[index]
            cropped_person = self._resize(self._crop_person(annotated_img, bounding_box))
            self._db_handler.insert_image(self._encode_image_to_jpg(cropped_person))
        # Отобразить рамку вокруг человека и её центр
        for i, (bbox, skeleton_detected) in enumerate(zip(skeletons_bounding_boxes, skeletons_detected)):
            if skeleton_detected:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            bbox_center = (
                bbox['min_x'] + int((bbox['max_x'] - bbox['min_x']) / 2),
                bbox['min_y'] + int((bbox['max_y'] - bbox['min_y']) / 2),
            )
            annotated_img = cv.rectangle(annotated_img,
                                         pt1=(bbox['min_x'], bbox['min_y']),
                                         pt2=(bbox['max_x'], bbox['max_y']),
                                         color=color, thickness=3)
            annotated_img = cv.circle(annotated_img,
                                      center=bbox_center,
                                      radius=5,
                                      color=color, thickness=3)
            annotated_img = cv.putText(annotated_img,
                                       text=f'Object: {i}',
                                       org=(bbox['max_x'], bbox['max_y']),
                                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=2)
        # Отобразить флаг момента поднятия рук
        for index in unique_raising_arms_moment_skeletons_idx:
            bounding_box = skeletons_bounding_boxes[index]
            annotated_img = cv.putText(annotated_img,
                                       text=f'Unique {index}',
                                       org=(bounding_box['min_x'], bounding_box['min_y']),
                                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=0, thickness=2)
        # Отобразить номер текущего кадра
        annotated_img = cv.putText(annotated_img,
                                   text=f'Current frame: {self._cur_frame}',
                                   org=(0, 30),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=0, thickness=2)
        return annotated_img

    def _reload_video(self):
        self._video_reader.release()
        self._video_reader = cv.VideoCapture(self._input_video_file)

    def run(self) -> None:
        while True:
            ret, frame_bgr = self._video_reader.read()
            if not ret:
                self._reload_video()
                continue
            self._cur_frame += 1
            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            processed_image = self._process_frame(frame_rgb)
            stream_frame = self._encode_image_to_jpg(processed_image)
            try:
                self._streamer.update(stream_frame)
            except TypeError:
                logging.debug('Server is not ready yet.')
