"""Build local submaps from published segment updates and observation poses."""

from copy import deepcopy
from dataclasses import dataclass
from typing import List

import numpy as np

from roman.map.map import Submap, SubmapParams, extract_submap_descriptors
from roman.object.segment import SegmentMinimalData


@dataclass
class SegmentUpdateResult:
    submap_created: bool
    submap_segments: List[SegmentMinimalData] = None


class SegmentQueue:
    def __init__(self, submap_num_segments: int, submap_overlapping_segments: int):
        if submap_num_segments < 1:
            raise ValueError("submap_num_segments must be positive")
        if not 0 <= submap_overlapping_segments < submap_num_segments:
            raise ValueError("submap_overlapping_segments must be in [0, submap_num_segments)")
        self.segments = {}
        self.submap_num_segments = submap_num_segments
        self.submap_overlapping_segments = submap_overlapping_segments

    def update(self, segment: SegmentMinimalData):
        existing = segment.id in self.segments
        self.segments[segment.id] = segment
        if existing or len(self.segments) < self.submap_num_segments:
            return SegmentUpdateResult(False)

        submap = list(self.segments.values())
        segments_sorted_by_time = sorted(submap, key=lambda seg: seg.reference_time())
        num_removed = self.submap_num_segments - self.submap_overlapping_segments
        for segment in segments_sorted_by_time[:num_removed]:
            del self.segments[segment.id]
        return SegmentUpdateResult(True, submap)

    @property
    def earliest_seen(self):
        return min((seg.first_seen for seg in self.segments.values()), default=None)


class SubmapBuilder:
    def __init__(self, submap_num_segments: int, submap_overlapping_segments: int,
                 params: SubmapParams):
        self.segment_queue = SegmentQueue(submap_num_segments, submap_overlapping_segments)
        self.params = params
        self.poses = {}
        self.descriptors = {}
        self.next_submap_id = 0

    def update_frame(self, time: float, pose_flu: np.ndarray, descriptor: np.ndarray = None):
        self.poses[time] = pose_flu.copy()
        if descriptor is not None:
            self.descriptors[time] = descriptor.copy()

    def update_segment(self, segment: SegmentMinimalData):
        if not self.poses:
            raise ValueError("A frame pose must be recorded before updating submap segments")
        result = self.segment_queue.update(segment)
        if not result.submap_created:
            return None

        reference_time = np.mean([seg.reference_time() for seg in result.submap_segments])
        reference_time = min(self.poses, key=lambda time: (abs(time - reference_time), time))
        submap = Submap(
            id=self.next_submap_id,
            time=reference_time,
            pose_flu=self.poses[reference_time].copy(),
            segments=deepcopy(result.submap_segments),
            segment_frame='submap_gravity_aligned',
        )
        # pose_gravity_aligned currently mutates its pose argument; keep the
        # original FLU pose so body-frame loop closure conversion retains tilt.
        T_center_odom = np.linalg.inv(submap.pose_gravity_aligned)
        for segment in submap.segments:
            segment.transform(T_center_odom)

        mode = self.params.submap_descriptor
        if mode == 'mean_semantic':
            descriptors = [seg.semantic_descriptor.flatten() for seg in submap.segments
                           if seg.semantic_descriptor is not None]
            if descriptors:
                descriptor = np.mean(descriptors, axis=0)
                norm = np.linalg.norm(descriptor)
                submap.descriptor = descriptor / norm if norm > 0 else descriptor
        elif mode is not None:
            times = sorted(time for time in self.descriptors
                           if submap.first_seen <= time <= submap.last_seen)
            if times:
                extract_submap_descriptors(
                    [submap], times, [self.descriptors[time] for time in times],
                    [self.poses[time] for time in times], self.params)

        self.next_submap_id += 1
        self._prune_frame_history()
        return submap

    def _prune_frame_history(self):
        times = sorted(self.poses)
        earliest_seen = self.segment_queue.earliest_seen
        if earliest_seen is None:
            keep_from = times[-1]
        else:
            index = max(0, int(np.searchsorted(times, earliest_seen)) - 1)
            keep_from = times[index]
        self.poses = {time: pose for time, pose in self.poses.items() if time >= keep_from}
        self.descriptors = {time: descriptor for time, descriptor in self.descriptors.items()
                            if time >= keep_from}
