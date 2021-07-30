import logging

import cv2 as cv
import numpy as np

from .database_handler.db_handler import DBHandler
from .detectors import SimpleRaisedArmsDetector, UniqueDetector
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
    def _resize(img: np.ndarray, max_dim_px: int = 100):
        factor = max_dim_px / max(img.shape)
        return cv.resize(img, dsize=(0, 0), fx=factor, fy=factor)

    @staticmethod
    def _crop_person(img: np.ndarray, skeleton: dict) -> np.ndarray:
        x = [point[0] for point in skeleton.values()]
        y = [point[1] for point in skeleton.values()]
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        return img[min_y: max_y, min_x: max_x]

    def __init__(self, input_video_file: str, db_name: str, db_user: str, db_password: str,
                 host: str = None, port: int = None, db_host: str = None, db_port: int = None):
        self._input_video_file = input_video_file
        self._detector = None
        self._video_reader = cv.VideoCapture(self._input_video_file)
        frame_shape = self._video_reader.read()[1].shape
        self._pose_estimator = PoseEstimator(frame_shape)
        self._pose_detector = SimpleRaisedArmsDetector()
        self._db_handler = DBHandler(db_name, db_user, db_password, db_host, db_port)
        # Эта шутка должна ловить момент поднятия рук и сбрасываться при их опускании
        self._unique_detector = UniqueDetector()
        self._streamer = Streamer(host, port, daemon=True)
        self._db_handler.connect()
        self._streamer.start()

    def _process_frame(self, img: np.ndarray) -> np.ndarray:
        poses, _ = self._pose_estimator.process_image(img)
        annotated_poses = self._pose_estimator.get_annotated_poses(poses)
        annotated_img = self._pose_estimator.draw_poses(img, poses)
        skeletons_detected = self._pose_detector.detect(annotated_poses)
        for skeleton in skeletons_detected:
            logging.debug('Raised arms were detected!')
            cropped_person = self._resize(self._crop_person(annotated_img, skeleton))
            self._db_handler.insert_image(self._encode_image_to_jpg(cropped_person))
        return annotated_img

    def _reload_video(self):
        # Возможно, костыльно, но пускай пока так.
        self._video_reader.release()
        self._video_reader = cv.VideoCapture(self._input_video_file)

    def run(self) -> None:
        while True:
            ret, frame_bgr = self._video_reader.read()
            if not ret:
                self._reload_video()
                continue
            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            processed_image = self._process_frame(frame_rgb)
            stream_frame = self._encode_image_to_jpg(processed_image)
            try:
                self._streamer.update(stream_frame)
            except TypeError:
                logging.debug('Server is not ready yet.')
