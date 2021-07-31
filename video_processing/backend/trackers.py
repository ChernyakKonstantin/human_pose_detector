from typing import List

import numpy as np

from .person import Person


class SkeletonTracker:
    """Трекер для отслеживания скелетов. Необходим, чтобы отслеживать, кто из людей в кадре принимал заданную позу."""
    _tracking_threshold = 0.5
    # Квадарт максимального сдвига элемента скелета (чтобы не считать sqrt)
    _shift_threshold = 100 ** 2
    _unmatched_frames_count = 10

    def __init__(self):
        self._skeletons: List[Person] = []

    def _match_skeletons(self, skeleton: dict, reference_skeleton: dict) -> bool:
        point_names = set(skeleton.keys()).intersection(set(reference_skeleton.keys()))
        tracks = np.array([np.square(skeleton[pt_name] - reference_skeleton[pt_name]).sum() for pt_name in point_names])
        if tracks.shape[0] != 0:
            return tracks[tracks <= self._shift_threshold].shape[0] / tracks.shape[0] >= self._tracking_threshold
        else:
            return False

    def _on_empty_intrenal_data(self, skeletons: List[dict], bboxes: List[dict]) -> None:
        for skeleton, bbox in zip(skeletons, bboxes):
            skeleton_data = Person(skeleton, bbox, self._unmatched_frames_count)
            self._skeletons.append(skeleton_data)

    def _on_empty_input_data(self):
        for reference_skeleton in self._skeletons:
            reference_skeleton.unmatched_frames_count -= 1
            if reference_skeleton.unmatched_frames_count == 0:
                self._skeletons.remove(reference_skeleton)

    def _on_extra_internal_data(self, skeletons: List[dict]) -> None:
        for reference_skeleton in self._skeletons:
            for skeleton in skeletons:
                if not self._match_skeletons(skeleton, reference_skeleton.skeleton):
                    reference_skeleton.unmatched_frames_count -= 1
                    if reference_skeleton.unmatched_frames_count == 0:
                        self._skeletons.remove(reference_skeleton)

    def _on_lack_internal_data(self, skeletons: List[dict], bboxes: List[dict]):
        new_skeletons = []
        for skeleton, bbox in zip(skeletons, bboxes):
            for reference_skeleton in self._skeletons:
                if not self._match_skeletons(skeleton, reference_skeleton.skeleton):
                    new_skeletons.append(Person(skeleton, bbox, self._unmatched_frames_count))
        self._skeletons.extend(new_skeletons)

    def update(self, skeletons: List[dict], bboxes: List[dict]) -> None:
        if len(self._skeletons) == 0:
            self._on_empty_intrenal_data(skeletons, bboxes)
        if len(skeletons) == 0:
            self._on_empty_input_data()
        elif len(self._skeletons) < len(skeletons):
            self._on_lack_internal_data(skeletons, bboxes)
        elif len(self._skeletons) > len(skeletons):
            self._on_extra_internal_data(skeletons)

    def track(self, skeletons: List[dict], bboxes: List[dict]) -> List[Person]:
        idx = []
        for skeleton, bbox in zip(skeletons, bboxes):
            for index, reference_skeleton in enumerate(self._skeletons):
                if self._match_skeletons(skeleton, reference_skeleton.skeleton):
                    self._skeletons[index].skeleton = skeleton
                    self._skeletons[index].bbox = bbox
                    idx.append(index)
        return [self._skeletons[index] for index in idx]
