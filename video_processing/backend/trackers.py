from typing import List

import numpy as np


class Skeleton:
    """Структура для хранения скелетов."""

    @property
    def unmatch_frames_count(self):
        return self._unmatch_frames_count

    @unmatch_frames_count.setter
    def unmatch_frames_count(self, value):
        self._unmatch_frames_count = value

    @property
    def skeleton(self):
        return self._skeleton

    @skeleton.setter
    def skeleton(self, value):
        self._skeleton = value

    def __init__(self, unmatch_frames_count, skeleton):
        self._unmatch_frames_count = unmatch_frames_count
        self._skeleton = skeleton


class SkeletonTracker:
    """Трекер для отслеживания скелетов. Необходим, чтобы отслеживать, кто из людей в кадре принимал заданную позу."""
    _tracking_threshold = 0.5
    # Квадарт максимального сдвига элемента скелета (чтобы не считать sqrt)
    _shift_threshold = 100 ** 2
    _unmatch_frames_count = 10

    def __init__(self):
        self._skeletons: List[Skeleton] = []

    def _match_skeletons(self, skeleton: dict, reference_skeleton: dict) -> bool:
        point_names = set(skeleton.keys()).intersection(set(reference_skeleton.keys()))
        tracks = np.array([np.square(skeleton[pt_name] - reference_skeleton[pt_name]).sum() for pt_name in point_names])
        return tracks[tracks <= self._shift_threshold].shape[0] / tracks.shape[0] >= self._tracking_threshold

    def _update(self, skeletons: List[dict]) -> None:
        if len(self._skeletons) == 0:
            self._skeletons.extend([Skeleton(self._unmatch_frames_count, skeleton) for skeleton in skeletons])
        elif len(self._skeletons) != len(skeletons):
            for skeleton in skeletons:
                for reference_skeleton in self._skeletons:
                    if not self._match_skeletons(skeleton, reference_skeleton.skeleton):
                        if reference_skeleton.unmatch_frames_count == 0:
                            if len(self._skeletons) > len(skeletons):
                                self._skeletons.append(Skeleton(self._unmatch_frames_count, skeleton))
                            else:
                                self._skeletons.remove(reference_skeleton)
                        else:
                            reference_skeleton.unmatch_frames_count -= 1
                    else:
                        reference_skeleton.unmatch_frames_count = self._unmatch_frames_count

    def track(self, skeletons: List[dict]) -> List[int]:
        self._update(skeletons)
        idx = []
        for skeleton in skeletons:
            for index, reference_skeleton in enumerate(self._skeletons):
                if self._match_skeletons(skeleton, reference_skeleton.skeleton):
                    self._skeletons[index].skeleton = skeleton
                    idx.append(index)
        return idx
