"""Robotiq 2F-85 gripper model: geometry table, software mimic references, grasp command."""
import math

import numpy as np
import pytest

from gp7_kinematics import GP7Kinematics, GripperModel


@pytest.fixture(scope='module')
def grp():
    return GripperModel.from_config()


def test_width_table(grp):
    # 85 mm Robotiq stroke, minus the 0.5 mm box pads on both sides
    assert math.isclose(grp.open_width, 0.084, abs_tol=5e-4)
    assert grp.width_for_theta(grp.theta_open) == pytest.approx(grp.open_width)
    assert np.all(np.diff(grp._width) < 0), 'pad gap must shrink monotonically while closing'
    for w in (0.08, 0.05, 0.02):
        assert grp.width_for_theta(grp.theta_for_width(w)) == pytest.approx(w, abs=1e-6)


def test_mimic_references(grp):
    assert grp.joint_names[0] == grp.driven_joint
    ref = dict(zip(grp.joint_names, grp.references(0.4)))
    assert ref['robotiq_85_left_knuckle_joint'] == pytest.approx(0.4)
    assert ref['robotiq_85_right_knuckle_joint'] == pytest.approx(-0.4)
    # pads parallel: knuckle + finger tip = 0 on each side
    assert ref['robotiq_85_left_knuckle_joint'] + ref['robotiq_85_left_finger_tip_joint'] == pytest.approx(0)
    assert ref['robotiq_85_right_knuckle_joint'] + ref['robotiq_85_right_finger_tip_joint'] == pytest.approx(0)
    assert grp.references(5.0)[0] == pytest.approx(grp.theta_closed)        # clamped


def test_grasp_command(grp):
    th, th_c = grp.grasp_theta(0.05, 20.0)
    assert grp.width_for_theta(th_c) == pytest.approx(0.05, abs=1e-6)
    # knuckle torque p * squeeze, acting through the lever, gives the grip force
    assert grp.knuckle_p * (th - th_c) / grp.lever == pytest.approx(20.0)
    refs = dict(zip(grp.joint_names, grp.references(th, th_c)))
    assert refs['robotiq_85_left_knuckle_joint'] == pytest.approx(th)
    assert refs['robotiq_85_left_finger_tip_joint'] == pytest.approx(-th_c)  # pads parallel at contact
    assert refs['robotiq_85_left_inner_knuckle_joint'] == pytest.approx(th_c)
    assert grp.contact_ok(th_c, 0.05) and not grp.contact_ok(grp.theta_closed, 0.05)


def test_tcp_geometry(grp):
    assert grp.pad_center_below_tcp(grp.reference_width) == pytest.approx(0.0, abs=1e-6)
    assert grp.pad_center_below_tcp(grp.open_width) < 0      # open pads ride higher on their arc
    # the pad runs down to the fingertip end: tips one pad half-height beyond the pad centre
    assert grp.tip_below_tcp(0.05) == pytest.approx(grp.pad_half_height, abs=1e-5)
    kin = GP7Kinematics.from_config(tool='tcp')
    assert kin.fk(np.zeros(6))[0, 3] == pytest.approx(0.56 + grp.tcp_z)


def test_move_time(grp):
    full = grp.move_time(grp.theta_open, grp.theta_closed)
    assert full == pytest.approx(grp.open_close_time)
    # cubic-spline peak speed 1.5 * dtheta / T within the URDF limit of 0.5 rad/s
    assert 1.5 * (grp.theta_closed - grp.theta_open) / full <= 0.5 + 1e-9
    assert grp.move_time(0.3, 0.35) >= 0.3
