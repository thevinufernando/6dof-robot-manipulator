"""Path, planner and full pick & place tests."""
import math
import os

import numpy as np
import pytest

from gp7_kinematics import GripperModel, top_down_pose
from gp7_motion.paths import CylindricalPath, LinePath
from gp7_motion.planner import PlanningError
from gp7_motion.sequence import load_config, make_planner, plan_pick_place

CFG = load_config(os.path.join(os.path.dirname(__file__), '..', 'config', 'pick_place.yaml'))
R_DOWN = top_down_pose(0, 0, 0)[:3, :3]


@pytest.fixture(scope='module')
def planner():
    return make_planner(CFG)


# --------------------------------------------------------------------------- paths
def test_line_path():
    p = LinePath([0.5, 0, 0.3], [0.5, 0, 0.1], R_DOWN)
    assert p.length == pytest.approx(0.2)
    assert np.allclose(p.position(0.1), [0.5, 0, 0.2])
    assert np.allclose(p.tangent(0.05), [0, 0, -1])
    assert np.allclose(p.pose(0.07)[:3, :3], R_DOWN)


def test_cylindrical_path_goes_through_the_front():
    p0, p1 = np.array([-0.5, -0.3, 0.35]), np.array([0.5, 0.5, 0.35])      # azimuth -149 -> +45 deg
    path = CylindricalPath(p0, p1, R_DOWN)
    s = np.linspace(0, path.length, 2001)
    pts = path.position(s)
    assert np.allclose(pts[0], p0) and np.allclose(pts[-1], p1, atol=1e-9)
    phi = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    assert np.all(np.abs(phi) < 170.0)                                      # never behind the robot
    assert np.allclose(np.linalg.norm(path.tangent(s), axis=1), 1.0)
    # tabulated arc length agrees with summing the chords
    assert np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) == pytest.approx(path.length, rel=1e-5)


# --------------------------------------------------------------------------- planner
def test_joint_move_is_synchronised(planner):
    q0, q1 = np.zeros(6), np.array([0.5, -0.4, 0.3, 0.2, -0.8, 1.0])
    tr = planner.plan_joint(q0, q1)
    assert np.allclose(tr.q[0], q0) and np.allclose(tr.q[-1], q1)
    sigma = (tr.q - q0) / (q1 - q0)                    # every joint at the same fraction
    assert np.allclose(sigma, sigma[:, :1], atol=1e-12)
    assert np.all(np.abs(tr.qd) <= planner.lim.v * (1 + 1e-9))
    assert np.all(np.abs(tr.qdd) <= planner.lim.a * (1 + 1e-9))


def test_cartesian_line_keeps_orientation_and_limits(planner):
    kin = planner.kin
    q0 = kin.ik_closest(top_down_pose(0.55, -0.3, 0.35)).q
    path = LinePath([0.55, -0.3, 0.35], [0.55, -0.3, 0.025], R_DOWN)
    tr = planner.plan_cartesian(path, 0.25, 0.5, 2.5, q0)
    assert np.max(tr.orientation_error) < 1e-7
    assert np.allclose(tr.tcp, path.position(tr.s[:, 0]), atol=1e-9)        # on the line
    assert np.all(np.abs(tr.qd) <= planner.lim.v + 1e-9)
    # Jacobian velocities agree with differentiating the IK positions
    qd_num = np.gradient(tr.q, tr.t, axis=0, edge_order=2)
    assert np.max(np.abs(qd_num - tr.qd)) < 2e-3


def test_cartesian_rejects_wrong_start(planner):
    with pytest.raises(PlanningError):
        planner.plan_cartesian(LinePath([0.5, 0, 0.3], [0.5, 0, 0.1], R_DOWN), 0.1, 0.5, 2.5, np.zeros(6))


def test_cartesian_slows_down_to_meet_joint_limits(planner):
    kin = planner.kin
    p0, p1 = [0.55, -0.3, 0.35], [0.5, 0.5, 0.35]
    q0 = kin.ik_closest(top_down_pose(*p0)).q
    tr = planner.plan_cartesian(CylindricalPath(p0, p1, R_DOWN), 5.0, 20.0, 200.0, q0)   # far too fast
    assert tr.stretch > 1.0
    assert np.all(np.abs(tr.qd) <= planner.lim.v + 1e-9)
    assert np.all(np.abs(tr.qdd) <= planner.lim.a + 1e-9)


# --------------------------------------------------------------------------- pick & place
def _check_plan(plan, planner):
    prev = None
    for st in plan.arm_steps:
        tr = st.trajectory
        if prev is not None:
            assert np.allclose(tr.q_start, prev, atol=1e-9), f'{tr.name}: discontinuous start'
        prev = tr.q_end
        assert np.all(np.abs(tr.qd) <= planner.lim.v + 1e-9), tr.name
        assert np.all(np.abs(tr.qdd) <= planner.lim.a * 1.001), tr.name
        assert np.all(tr.q >= planner.lim.lower - 1e-9) and np.all(tr.q <= planner.lim.upper + 1e-9)
        if st.carrying:   # Box B upright: tool axis vertical and orientation unchanged
            assert tr.kind == 'cartesian'
            assert np.max(tr.orientation_error) < 1e-7, tr.name


@pytest.mark.parametrize('profile', ['scurve', 'trapezoidal'])
def test_pick_place_default(planner, profile):
    plan = plan_pick_place(planner, CFG, (0.55, -0.30, 0.0, 0.0), np.zeros(6), profile=profile)
    _check_plan(plan, planner)
    names = [s.name for s in plan.steps]
    assert names == ['open', 'to_ready', 'to_above_b', 'lower', 'approach', 'grasp', 'lift', 'transfer',
                     'descend', 'release', 'retreat', 'to_ready_end']
    descend = next(s.trajectory for s in plan.steps if s.name == 'descend')
    # 2F-85 TCP = pad centre at the 50 mm grasp -> Box B's mid-height is at the TCP
    assert np.allclose(descend.tcp[-1], [0.5, 0.5, 0.20 + 0.025 + 0.002], atol=1e-6)
    grasp = next(s for s in plan.steps if s.name == 'grasp')
    assert grasp.theta > grasp.theta_linkage                     # knuckles squeeze past the contact
    assert abs(grasp.width - 0.05) < 1e-12
    assert all(s.duration > 0 for s in plan.steps)


def test_free_moves_start_and_end_at_safe_height(planner):
    plan = plan_pick_place(planner, CFG, (0.192, 0.611, 0.0, 0.0), np.zeros(6))   # the Gazebo failure case
    for st in plan.arm_steps:
        if st.trajectory.kind == 'joint' and st.name != 'to_ready':
            assert st.trajectory.tcp[0, 2] >= CFG['task']['z_safe'] - 1e-9
            assert st.trajectory.tcp[-1, 2] >= CFG['task']['z_safe'] - 1e-9


def test_clearance_check_rejects_a_path_through_box_a(planner):
    from gp7_motion.sequence import check_box_a_clearance
    kin = planner.kin
    q0 = kin.ik_closest(top_down_pose(0.45, 0.0, 0.40)).q
    q1 = kin.ik_closest(top_down_pose(0.20, 0.61, 0.175), q0).q        # the old, too low pre-grasp
    with pytest.raises(PlanningError):
        check_box_a_clearance(planner.plan_joint(q0, q1, 'old_to_pregrasp'), CFG['task'], kin,
                              GripperModel.from_config())


def test_clearance_uses_the_turned_gripper_footprint(planner):
    """Box B 103 mm left of Box A's edge. The open 2F-85 reaches 76 mm along its closing
    axis (tool Y) but only 38 mm across it, so the grasp fits next to Box A only when the
    fingers do not point at it (needs 30 mm clearance)."""
    from gp7_motion.sequence import check_box_a_clearance, grasp_height
    kin, grp = planner.kin, GripperModel.from_config()
    x, y = 0.5 - 0.15 - 0.103, 0.5
    z = grasp_height(grp, 0.0, 0.05)
    q0 = kin.ik_closest(top_down_pose(x, y, 0.30)).q

    def approach(yaw):
        R = top_down_pose(0, 0, 0, yaw=yaw)[:3, :3]
        qs = kin.ik_closest(top_down_pose(x, y, 0.30, yaw=yaw), q0).q
        return planner.plan_cartesian(LinePath([x, y, 0.30], [x, y, z], R), 0.1, 0.5, 2.5, qs, 'approach')

    check_box_a_clearance(approach(0.0), CFG['task'], kin, grp)          # closes along world y: fits
    with pytest.raises(PlanningError):                                   # closes along world x: hits
        check_box_a_clearance(approach(math.pi / 2), CFG['task'], kin, grp)


def test_grasp_height_keeps_fingertips_off_the_floor():
    from gp7_motion.sequence import FLOOR_CLEARANCE, grasp_height
    grp = GripperModel.from_config()
    assert math.isclose(grasp_height(grp, 0.0, 0.05), 0.025, abs_tol=1e-4)    # pad centre = box centre
    for size in (0.02, 0.03, 0.05, 0.07):
        z = grasp_height(grp, 0.0, size)
        assert z - grp.tip_below_tcp(size) >= FLOOR_CLEARANCE - 1e-12


def test_pick_place_random_box_b(planner):
    """The whole task is feasible for random Box B poses in the launch file's pick region."""
    rng = np.random.default_rng(11)
    n = 0
    while n < 40:
        r = math.sqrt(rng.uniform(0.40 ** 2, 0.75 ** 2))
        a = math.radians(rng.uniform(-150, 150))
        x, y = r * math.cos(a), r * math.sin(a)
        if max(abs(x - 0.5), abs(y - 0.5)) <= 0.25:          # keep off Box A, as the launch does
            continue
        plan = plan_pick_place(planner, CFG, (x, y, 0.0, rng.uniform(-math.pi, math.pi)), np.zeros(6))
        _check_plan(plan, planner)
        n += 1
