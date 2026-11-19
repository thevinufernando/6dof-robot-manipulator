"""Tests for the GP7 forward / inverse kinematics.

Run:  cd ros2_ws && colcon test --packages-select gp7_kinematics --event-handlers console_direct+
  or  python3 -m pytest ros2_ws/src/gp7_kinematics/test -q
"""
import math

import numpy as np
import pytest

from gp7_kinematics import GP7Kinematics, GripperModel, pose_error, top_down_pose


@pytest.fixture(scope='module')
def tcp():
    return GP7Kinematics.from_config(tool='tcp')


@pytest.fixture(scope='module')
def flange():
    return GP7Kinematics.from_config(tool='flange')


def random_q(kin, rng, n, margin=0.05, avoid_wrist_singularity=True):
    out = []
    while len(out) < n:
        q = rng.uniform(kin.lower + margin, kin.upper - margin)
        if avoid_wrist_singularity and abs(math.sin(q[4])) < 0.05:
            continue
        out.append(q)
    return out


# --------------------------------------------------------------------------- forward kinematics
def test_fk_home(flange, tcp):
    # Flange at (a1 + d4 + d6, 0, d1 + a2 + a3), z out of the flange along +X0.
    T = flange.fk(np.zeros(6))
    assert np.allclose(T[:3, 3], [0.56, 0.0, 0.815])
    assert np.allclose(T[:3, 2], [1.0, 0.0, 0.0])
    # Robotiq 2F-85 TCP: 12 mm coupling + pad centre at the 50 mm grasp (gripper_geometry.yaml)
    assert np.allclose(tcp.fk(np.zeros(6))[:3, 3], [0.56 + GripperModel.from_config().tcp_z, 0.0, 0.815])


def test_fk_matches_gazebo(flange):
    # Flange pose measured in Gazebo / TF for q = (0, -0.5, 0.3, 0, 0.4, 0).
    T = flange.fk([0.0, -0.5, 0.3, 0.0, 0.4, 0.0])
    assert np.allclose(T[:3, 3], [0.770926, 0.0, 0.688206], atol=2e-6)


# --------------------------------------------------------------------------- inverse kinematics
@pytest.mark.parametrize('tool', ['tcp', 'flange'])
def test_ik_round_trip(tool):
    """FK(q) -> IK -> the original q is among the solutions; every solution reproduces the pose."""
    kin = GP7Kinematics.from_config(tool=tool)
    rng = np.random.default_rng(42)
    for q in random_q(kin, rng, 300):
        T = kin.fk(q)
        sols = kin.ik(T, q_ref=q)
        assert sols, f'no solution for reachable q={q}'
        assert np.allclose(sols[0].q, q, atol=1e-7), 'closest solution must be the original q'
        for s in sols:
            dp, dr = pose_error(kin.fk(s.q), T)
            assert dp < 1e-7 and dr < 1e-7
            assert kin.within_limits(s.q)


def test_eight_solutions_without_limits(flange):
    T = flange.fk([0.3, -0.4, 0.2, 0.5, -0.8, 0.4])
    sols = flange.ik(T, respect_limits=False)
    assert len(sols) == 8
    assert {(s.shoulder, s.elbow, s.wrist) for s in sols} == {
        (sh, el, wr) for sh in ('front', 'back') for el in ('up', 'down') for wr in ('B+', 'B-')}


def test_wrist_singularity(tcp):
    q = np.array([0.2, -0.3, 0.4, 0.7, 0.0, -0.2])      # q5 = 0: only q4 + q6 is defined
    T = tcp.fk(q)
    sol = tcp.ik_closest(T, q_ref=q)
    assert sol.wrist == 'singular'
    dp, dr = pose_error(tcp.fk(sol.q), T)
    assert dp < 1e-7 and dr < 1e-7
    assert math.isclose(sol.q[3] + sol.q[5], q[3] + q[5], abs_tol=1e-7)


def test_unreachable(tcp):
    assert tcp.ik(top_down_pose(1.5, 0.0, 0.3)) == []


def test_top_down_grasp_keeps_box_upright(tcp):
    """Top-down poses (approach = -Z0) used for picking: reachable, tool axis exactly vertical."""
    rng = np.random.default_rng(7)
    for _ in range(100):
        r, a = rng.uniform(0.40, 0.75), rng.uniform(-math.radians(150), math.radians(150))
        T = top_down_pose(r * math.cos(a), r * math.sin(a), 0.025, yaw=rng.uniform(-1, 1))
        sol = tcp.ik_closest(T)
        T_fk = tcp.fk(sol.q)
        assert np.allclose(T_fk[:3, 2], [0.0, 0.0, -1.0], atol=1e-9)
        assert np.allclose(T_fk[:3, 3], T[:3, 3], atol=1e-9)


def test_task_box_a_reachability(tcp):
    # Box B placed on Box A's top centre (z = 0.20 + 0.025): unreachable at (1, 1).
    assert tcp.ik(top_down_pose(1.0, 1.0, 0.225)) == []
    assert tcp.ik(top_down_pose(0.6, 0.6, 0.225)) != []


# --------------------------------------------------------------------------- Jacobian
def _rot_log(R):
    """Rotation vector of a (small) rotation matrix."""
    a = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    if a < 1e-12:
        return np.zeros(3)
    return a / (2 * math.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


@pytest.mark.parametrize('tool', ['tcp', 'flange'])
def test_jacobian_matches_finite_differences(tool):
    kin = GP7Kinematics.from_config(tool=tool)
    rng = np.random.default_rng(3)
    h = 1e-6
    for q in random_q(kin, rng, 50):
        J = kin.jacobian(q)
        T0 = kin.fk(q)
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = h
            T1 = kin.fk(q + dq)
            v = (T1[:3, 3] - T0[:3, 3]) / h
            w = _rot_log(T1[:3, :3] @ T0[:3, :3].T) / h
            assert np.allclose(J[:3, i], v, atol=1e-5)
            assert np.allclose(J[3:, i], w, atol=1e-5)


def test_joint_velocities_and_singularity(tcp):
    q = np.array([0.3, -0.5, -0.2, 0.4, -0.9, 0.2])
    v = np.array([0.1, -0.05, 0.02])
    qd = tcp.joint_velocities(q, v)
    assert np.allclose(tcp.jacobian(q) @ qd, np.r_[v, 0, 0, 0])
    assert tcp.manipulability(q) > 1e-3
    assert tcp.manipulability([0.3, -0.5, -0.2, 0.4, 0.0, 0.2]) < 1e-7   # wrist singular (q5 = 0): ~0 up to round-off
