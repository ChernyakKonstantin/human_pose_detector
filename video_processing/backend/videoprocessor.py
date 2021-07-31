import logging
from typing import List

import cv2 as cv
import numpy as np

from .database_handler.db_handler import DBHandler
from .detectors import RaisingArmsMomentDetector, SimpleRaisedArmsDetector
from .person import Person
from .pose_estimator import PoseEstimator
from .streamer import Streamer
from .trackers import SkeletonTracker

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

    def _draw_info(self, img: np.ndarray, skeletons_data: List[Person],
                   poses_detected: List[bool], unique_poses: List[bool]) -> np.ndarray:
        # Отобразить флаг измененения позы
        for index, unique_pose_flag in enumerate(unique_poses):
            if unique_pose_flag:
                bbox = skeletons_data[index].bbox
                img = cv.putText(img,
                                 text=f'Unique {index}',
                                 org=(bbox['min_x'], bbox['min_y']),
                                 fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=0, thickness=2)
        # Отобразить рамку вокруг человека и её центр
        for index, (pose_detected_flag, skeleton_data) in enumerate(zip(poses_detected, skeletons_data)):
            if pose_detected_flag:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            bbox = skeleton_data.bbox
            bbox_center = (
                bbox['min_x'] + int((bbox['max_x'] - bbox['min_x']) / 2),
                bbox['min_y'] + int((bbox['max_y'] - bbox['min_y']) / 2),
            )
            img = cv.rectangle(img,
                               pt1=(bbox['min_x'], bbox['min_y']),
                               pt2=(bbox['max_x'], bbox['max_y']),
                               color=color, thickness=2)
            img = cv.circle(img,
                            center=bbox_center,
                            radius=3,
                            color=color, thickness=2)
            img = cv.putText(img,
                             text=f'Object: {index}',
                             org=(bbox['max_x'], bbox['max_y']),
                             fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=0, thickness=2)
        # Отобразить номер текущего кадра
        img = cv.putText(img,
                         text=f'Current frame: {self._cur_frame}',
                         org=(0, 30),
                         fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=0, thickness=2)
        return img

    def _process_frame(self, img: np.ndarray) -> np.ndarray:
        skeletons, _ = self._pose_estimator.process_image(img)
        annotated_img = self._pose_estimator.draw_poses(img, skeletons)
        annotated_skeletons = self._pose_estimator.annotate_skeletons(skeletons)
        skeletons_bounding_boxes = self.get_bounding_boxes(annotated_skeletons)
        self._skeleton_tracker.update(annotated_skeletons, skeletons_bounding_boxes)
        skeletons_data = self._skeleton_tracker.track(annotated_skeletons, skeletons_bounding_boxes)
        poses_detected = self._pose_detector.detect([skeleton_data.skeleton for skeleton_data in skeletons_data])
        unique_poses = self._unique_detector.detect(skeletons_data, poses_detected)
        for index, unique_pose_flag in enumerate(unique_poses):
            if unique_pose_flag:
                logging.debug(f'Raised arms were detected on skeleton: {index}!')
                bbox = skeletons_data[index].bbox
                cropped_person = self._resize(self._crop_person(annotated_img, bbox))
                self._db_handler.insert_image(self._encode_image_to_jpg(cropped_person))
        annotated_img = self._draw_info(annotated_img, skeletons_data, poses_detected, unique_poses)
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
