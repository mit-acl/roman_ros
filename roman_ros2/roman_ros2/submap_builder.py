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


class NoKeyframeError(Exception):
    """A submap was completed but no keyframe lies within its segments' time span."""


class SubmapBuilder:
    def __init__(self, submap_num_segments: int, submap_overlapping_segments: int,
                 params: SubmapParams, reference_by_keyframe: bool = False):
        self.segment_queue = SegmentQueue(submap_num_segments, submap_overlapping_segments)
        self.params = params
        self.reference_by_keyframe = reference_by_keyframe
        self.poses = {}
        self.descriptors = {}
        self.keyframes = {}
        self.next_submap_id = 0

    def update_frame(self, time: float, pose_flu: np.ndarray, descriptor: np.ndarray = None):
        self.poses[time] = pose_flu.copy()
        if descriptor is not None:
            self.descriptors[time] = descriptor.copy()

    def update_keyframe(self, time: float, pose_flu: np.ndarray):
        """Record an externally chosen keyframe (e.g. a pose graph node) to reference submaps to."""
        self.keyframes[time] = pose_flu.copy()

    def update_segment(self, segment: SegmentMinimalData):
        if not self.poses:
            raise ValueError("A frame pose must be recorded before updating submap segments")
        result = self.segment_queue.update(segment)
        if not result.submap_created:
            return None

        reference_time = np.mean([seg.reference_time() for seg in result.submap_segments])
        if self.reference_by_keyframe:
            # only keyframes the submap's segments were observed around may anchor it
            first_seen = min(seg.first_seen for seg in result.submap_segments)
            last_seen = max(seg.last_seen for seg in result.submap_segments)
            times = [time for time in self.keyframes if first_seen <= time <= last_seen]
            if not times:
                self._prune_frame_history()
                raise NoKeyframeError(
                    f"No keyframe in submap time span [{first_seen:.3f}, {last_seen:.3f}]")
            reference_time = min(times, key=lambda time: (abs(time - reference_time), time))
            pose_flu = self.keyframes[reference_time]
        else:
            reference_time = min(self.poses, key=lambda time: (abs(time - reference_time), time))
            pose_flu = self.poses[reference_time]
        submap = Submap(
            id=self.next_submap_id,
            time=reference_time,
            pose_flu=pose_flu.copy(),
            segments=deepcopy(result.submap_segments),
            segment_frame='submap_gravity_aligned',
        )
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
        earliest_seen = self.segment_queue.earliest_seen
        keep_from = self._keep_from(self.poses, earliest_seen)
        self.poses = {time: pose for time, pose in self.poses.items() if time >= keep_from}
        self.descriptors = {time: descriptor for time, descriptor in self.descriptors.items()
                            if time >= keep_from}
        if self.keyframes:
            keep_from = self._keep_from(self.keyframes, earliest_seen)
            self.keyframes = {time: keyframe for time, keyframe in self.keyframes.items()
                              if time >= keep_from}

    @staticmethod
    def _keep_from(history, earliest_seen):
        times = sorted(history)
        if earliest_seen is None:
            return times[-1]
        index = max(0, int(np.searchsorted(times, earliest_seen)) - 1)
        return times[index]
