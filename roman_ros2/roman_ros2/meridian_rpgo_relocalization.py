"""
Relocalization against fixed prior maps with Meridian's CLIPPER loop closure consistency
invariant, as a replacement for Hydra-Multi's map <- odom estimate.

The prior maps are treated as a fixed frame (like Meridian's aerial image): a loop closure
between a live submap (in the live robot's odom frame) and a prior submap (in the map frame)
is one candidate estimate of T_map_odom. CLIPPER picks the largest mutually consistent set of
candidates, with the consistency tolerance growing with the live path length between them, and
the inliers are fused either by averaging (frame_align) or by a pose graph over the live
odometry keyframes (pgo).

This module has no ROS dependency. meridian (clipperpy) and gtsam are imported lazily: they are
not ROMAN dependencies and are only needed when this relocalizer is used.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot


@dataclass
class MeridianRPGOParams:
    # CLIPPER consistency (see clipper/invariants/loop_closure_consistency.h)
    rot_consistency_sigma_deg: float = 5.0
    rot_consistency_eps_deg: float = 5.0
    trans_consistency_sigma_m: float = 2.0
    trans_consistency_eps_m: float = 2.0
    added_trans_noise_m_per_m: float = 0.05
    added_rot_noise_deg_per_m: float = 0.02
    single_lc_per_live_submap: bool = True
    # weight each loop closure by min(1, num_associations / lc_score_associations_ref)
    fuse_lc_score: bool = False
    lc_score_associations_ref: float = 20.0
    # do not localize until CLIPPER finds at least this many inliers
    min_num_inliers: int = 4
    # "frame_align" (average of inlier T_map_odom) or "pgo" (pose graph over odom keyframes)
    optimization_method: str = "pgo"
    # pgo noise, per odometry keyframe step (rotation, translation)
    odom_rot_sigma_deg: float = 0.05
    odom_trans_sigma_m: float = 0.05
    prior_rot_sigma_deg: float = 5.0
    prior_trans_sigma_m: float = 2.0


@dataclass
class LoopClosureCandidate:
    live_submap_id: int
    prior_key: Tuple[int, int]      # (prior robot id, prior submap id)
    live_time: float                # live submap time (s)
    T_odom_live: np.ndarray         # live submap in the live odom frame
    T_map_prior: np.ndarray         # prior submap in the (fixed) map frame
    T_prior_live: np.ndarray        # registration: prior submap <- live submap
    num_associations: int

    @property
    def T_map_odom(self) -> np.ndarray:
        return self.T_map_prior @ self.T_prior_live @ inv(self.T_odom_live)


@dataclass
class RelocalizationResult:
    localized: bool
    num_candidates: int
    inlier_indices: np.ndarray
    T_map_odom: Optional[np.ndarray] = None
    # optimized keyframe poses (T_map_base) and their times, pgo only
    keyframe_times: Optional[np.ndarray] = None
    T_map_keyframes: Optional[List[np.ndarray]] = None


class OdometryKeyframes:
    """Live odometry history: T_odom_base keyframes and cumulative path length."""

    def __init__(self, min_trans_m: float = 0.5, min_rot_deg: float = 10.0):
        self.min_trans_m = min_trans_m
        self.min_rot_rad = np.deg2rad(min_rot_deg)
        self.times: List[float] = []
        self.poses: List[np.ndarray] = []
        self.path_lengths: List[float] = []

    def __len__(self):
        return len(self.times)

    def copy(self) -> "OdometryKeyframes":
        odometry = OdometryKeyframes(self.min_trans_m)
        odometry.min_rot_rad = self.min_rot_rad
        odometry.times = list(self.times)
        odometry.poses = list(self.poses)
        odometry.path_lengths = list(self.path_lengths)
        return odometry

    def add(self, t: float, T_odom_base: np.ndarray) -> bool:
        """Add a pose if it moved enough since the last keyframe; returns whether it was added."""
        if self.times:
            if t <= self.times[-1]:
                return False
            dT = inv(self.poses[-1]) @ T_odom_base
            step = np.linalg.norm(dT[:3, 3])
            angle = Rot.from_matrix(dT[:3, :3]).magnitude()
            if step < self.min_trans_m and angle < self.min_rot_rad:
                return False
            path_length = self.path_lengths[-1] + step
        else:
            path_length = 0.0
        self.times.append(t)
        self.poses.append(T_odom_base.copy())
        self.path_lengths.append(path_length)
        return True

    def path_length_at(self, t: float) -> float:
        if not self.times:
            return 0.0
        return float(np.interp(t, self.times, self.path_lengths))

    def nearest_index(self, t: float) -> int:
        return int(np.argmin(np.abs(np.asarray(self.times) - t)))


def average_poses(poses: List[np.ndarray]) -> np.ndarray:
    """Mean translation and chordal mean rotation of 4x4 poses."""
    T = np.eye(4)
    T[:3, :3] = Rot.from_matrix(np.array([P[:3, :3] for P in poses])).mean().as_matrix()
    T[:3, 3] = np.mean([P[:3, 3] for P in poses], axis=0)
    return T


class MeridianRPGORelocalizer:

    def __init__(self, params: MeridianRPGOParams):
        self.params = params
        assert params.optimization_method in ("frame_align", "pgo"), \
            f"unknown optimization_method: {params.optimization_method}"

    def lc_scores(self, candidates: List[LoopClosureCandidate]) -> np.ndarray:
        ref = max(self.params.lc_score_associations_ref, 1e-12)
        return np.clip([c.num_associations / ref for c in candidates], 0.0, 1.0)

    def run_clipper(self, candidates: List[LoopClosureCandidate],
                    odometry: OdometryKeyframes) -> np.ndarray:
        """Indices of the largest consistent set of candidates."""
        import clipperpy

        prior_idx: Dict[Tuple[int, int], int] = {}
        live_idx: Dict[int, int] = {}
        prior_poses, live_poses, live_path_lengths = [], [], []
        scores = self.lc_scores(candidates)
        D = np.zeros((len(candidates), 19))
        for k, c in enumerate(candidates):
            if c.prior_key not in prior_idx:
                prior_idx[c.prior_key] = len(prior_poses)
                prior_poses.append(c.T_map_prior)
            if c.live_submap_id not in live_idx:
                live_idx[c.live_submap_id] = len(live_poses)
                live_poses.append(c.T_odom_live)
                live_path_lengths.append(odometry.path_length_at(c.live_time))
            D[k, 0] = prior_idx[c.prior_key]
            D[k, 1] = live_idx[c.live_submap_id]
            D[k, 2:18] = c.T_prior_live.flatten()
            D[k, 18] = scores[k]

        p = self.params
        iparams = clipperpy.invariants.LoopClosureConsistencyParams()
        iparams.rot_sigma_rad = np.deg2rad(p.rot_consistency_sigma_deg)
        iparams.rot_eps_rad = np.deg2rad(p.rot_consistency_eps_deg)
        iparams.trans_sigma_m = p.trans_consistency_sigma_m
        iparams.trans_eps_m = p.trans_consistency_eps_m
        iparams.added_trans_noise_m_per_m = p.added_trans_noise_m_per_m
        iparams.added_rot_noise_deg_per_m = p.added_rot_noise_deg_per_m
        iparams.single_lc_per_ground_sm = p.single_lc_per_live_submap
        iparams.single_lc_per_ground_aerial_pair = True
        iparams.fuse_lc_score = p.fuse_lc_score
        invariant = clipperpy.invariants.LoopClosureConsistency(
            prior_poses, live_poses, live_path_lengths, iparams)
        clipper = clipperpy.CLIPPERPairwiseAndSingle(invariant, clipperpy.Params())
        # columns are data (Eigen column-major)
        D = D.T
        A = np.stack([np.arange(D.shape[1])] * 2, axis=1).astype(np.int32)
        clipper.score_pairwise_and_single_consistency(D, D, A)
        clipper.solve()
        return np.sort(np.array(clipper.get_solution().nodes, dtype=int))

    def solve(self, candidates: List[LoopClosureCandidate],
              odometry: OdometryKeyframes) -> RelocalizationResult:
        if len(candidates) < self.params.min_num_inliers:
            return RelocalizationResult(False, len(candidates), np.array([], dtype=int))

        inliers = self.run_clipper(candidates, odometry)
        if len(inliers) < self.params.min_num_inliers:
            return RelocalizationResult(False, len(candidates), inliers)

        T_map_odom = average_poses([candidates[i].T_map_odom for i in inliers])
        result = RelocalizationResult(True, len(candidates), inliers, T_map_odom)
        if self.params.optimization_method == "pgo" and len(odometry) >= 2:
            result.keyframe_times = np.asarray(odometry.times)
            result.T_map_keyframes = self.pgo([candidates[i] for i in inliers], odometry,
                                              T_map_odom)
            # map <- odom at the latest keyframe, so the published correction is current
            result.T_map_odom = result.T_map_keyframes[-1] @ inv(odometry.poses[-1])
        return result

    def pgo(self, inliers: List[LoopClosureCandidate], odometry: OdometryKeyframes,
            T_map_odom_init: np.ndarray) -> List[np.ndarray]:
        """Pose3 graph: odometry between keyframes plus one map prior per inlier."""
        import gtsam

        p = self.params
        # gtsam Pose3 tangent order: rotation, then translation
        odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [np.deg2rad(p.odom_rot_sigma_deg)] * 3 + [p.odom_trans_sigma_m] * 3))
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [np.deg2rad(p.prior_rot_sigma_deg)] * 3 + [p.prior_trans_sigma_m] * 3))

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        poses = odometry.poses
        for i, T_odom_base in enumerate(poses):
            initial.insert(i, gtsam.Pose3(T_map_odom_init @ T_odom_base))
        for i in range(len(poses) - 1):
            graph.add(gtsam.BetweenFactorPose3(
                i, i + 1, gtsam.Pose3(inv(poses[i]) @ poses[i + 1]), odom_noise))
        for c in inliers:
            i = odometry.nearest_index(c.live_time)
            # this loop closure's map <- odom applied to the keyframe nearest the live submap
            graph.add(gtsam.PriorFactorPose3(
                i, gtsam.Pose3(c.T_map_odom @ poses[i]), prior_noise))

        result = gtsam.LevenbergMarquardtOptimizer(graph, initial,
                                                   gtsam.LevenbergMarquardtParams()).optimize()
        return [result.atPose3(i).matrix() for i in range(len(poses))]
