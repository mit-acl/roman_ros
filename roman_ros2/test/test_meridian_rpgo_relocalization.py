import numpy as np
import pytest
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot

pytest.importorskip("clipperpy")
pytest.importorskip("gtsam")

from roman_ros2.meridian_rpgo_relocalization import (  # noqa: E402
    LoopClosureCandidate,
    MeridianRPGOParams,
    MeridianRPGORelocalizer,
    OdometryKeyframes,
)


def pose(x, y, z=0.0, yaw_deg=0.0):
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler("z", yaw_deg, degrees=True).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


# true map <- odom of the live robot
T_MAP_ODOM = pose(30.0, -12.0, 1.5, 70.0)


def make_problem(n_inliers, n_outliers, seed=0):
    """A live robot driving a loop, with loop closures to prior submaps every 10 m."""
    rng = np.random.default_rng(seed)
    odom = OdometryKeyframes(min_trans_m=0.5)
    times = np.arange(0, 400, 0.5)
    for t in times:
        a = t / 400 * 2 * np.pi
        odom.add(t, pose(50 * np.cos(a), 50 * np.sin(a), 0.0, np.degrees(a) + 90))

    candidates = []
    live_times = np.linspace(5, 395, n_inliers + n_outliers)
    rng.shuffle(live_times)
    for k, t in enumerate(live_times):
        T_odom_live = odom.poses[odom.nearest_index(t)]
        # prior submap near the live submap, in the map frame
        T_live_prior = pose(*rng.uniform(-5, 5, 2), 0.0, rng.uniform(-30, 30))
        T_map_prior = T_MAP_ODOM @ T_odom_live @ T_live_prior
        T_prior_live = inv(T_live_prior)
        if k >= n_inliers:
            # wrong registration: a different place, random offset
            T_prior_live = pose(*rng.uniform(-40, 40, 2), 0.0, rng.uniform(-180, 180))
        else:
            T_prior_live = T_prior_live @ pose(*rng.normal(0, 0.1, 2), 0.0, rng.normal(0, 0.5))
        candidates.append(LoopClosureCandidate(
            live_submap_id=k, prior_key=(1, k), live_time=float(t), T_odom_live=T_odom_live,
            T_map_prior=T_map_prior, T_prior_live=T_prior_live, num_associations=10))
    return candidates, odom


@pytest.mark.parametrize("method", ["frame_align", "pgo"])
def test_relocalizes_with_mostly_outliers(method):
    candidates, odom = make_problem(n_inliers=15, n_outliers=85)
    reloc = MeridianRPGORelocalizer(MeridianRPGOParams(optimization_method=method))
    result = reloc.solve(candidates, odom)

    assert result.localized
    assert set(result.inlier_indices) <= set(range(15))
    assert len(result.inlier_indices) >= 12
    err = inv(T_MAP_ODOM) @ result.T_map_odom
    assert np.linalg.norm(err[:3, 3]) < 0.3
    assert np.degrees(Rot.from_matrix(err[:3, :3]).magnitude()) < 1.0
    if method == "pgo":
        assert len(result.T_map_keyframes) == len(odom)


def test_not_localized_below_min_inliers():
    candidates, odom = make_problem(n_inliers=3, n_outliers=20)
    result = MeridianRPGORelocalizer(MeridianRPGOParams(min_num_inliers=4)).solve(
        candidates, odom)
    assert not result.localized
    assert result.T_map_odom is None
