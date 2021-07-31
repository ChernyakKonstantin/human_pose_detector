import numpy as np


class Person:
    """Структура для хранения данных о человеке в кадре."""

    @property
    def unmatched_frames_count(self):
        return self._unmatched_frames_count

    @unmatched_frames_count.setter
    def unmatched_frames_count(self, value):
        self._unmatched_frames_count = value

    @property
    def no_detection_count(self):
        return self._no_detection_count

    @no_detection_count.setter
    def no_detection_count(self, value):
        self._no_detection_count = value

    @property
    def skeleton(self):
        return self._skeleton

    @skeleton.setter
    def skeleton(self, value):
        self._skeleton = value

    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, value):
        self._bbox = value

    @property
    def unique_pose(self):
        return self._unique_pose

    @unique_pose.setter
    def unique_pose(self, value):
        self._unique_pose = value

    def __init__(self, skeleton: dict, bbox: dict, _unmatched_frames_count: int = None,
                 no_detection_count: int = None):
        self._unmatched_frames_count = _unmatched_frames_count
        self._no_detection_count = no_detection_count
        self._skeleton = skeleton
        self._bbox = bbox
        self._unique_pose = True
