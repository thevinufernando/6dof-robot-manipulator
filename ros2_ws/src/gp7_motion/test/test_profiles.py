"""Velocity profile tests: boundary conditions, limits, continuity, consistency."""
import numpy as np
import pytest

from gp7_motion.profiles import SCurveProfile, TrapezoidalProfile, make_profile

CASES = [  # D, v, a, j  -> covers: full profile, v not reached, a not reached, tiny move
    (0.50, 0.25, 0.5, 2.5),
    (0.05, 0.25, 0.5, 2.5),
    (0.002, 0.25, 0.5, 2.5),
    (1.00, 0.10, 2.0, 1.0),    # v*j < a^2: a_max never reached
    (3.00, 1.00, 0.8, 4.0),
]


def _check(prof, D, v, a, j=None):
    T = prof.duration
    t = np.linspace(0.0, T, 4001)
    s, sd, sdd, sddd = prof.sample(t)
    assert s[0] == pytest.approx(0.0, abs=1e-12) and s[-1] == pytest.approx(D, rel=1e-9, abs=1e-12)
    assert sd[0] == pytest.approx(0.0, abs=1e-12) and sd[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(s) >= -1e-12)                       # never moves backwards
    assert np.max(sd) <= v * (1 + 1e-9)
    assert np.max(np.abs(sdd)) <= a * (1 + 1e-9)
    if j is not None:
        assert np.max(np.abs(sddd)) <= j * (1 + 1e-9)
        assert sdd[0] == pytest.approx(0.0, abs=1e-12)      # S-curve: acceleration starts at 0
    # derivatives are consistent (numerical integration of s_dot gives s)
    s_int = np.concatenate([[0.0], np.cumsum(0.5 * (sd[1:] + sd[:-1]) * np.diff(t))])
    assert np.allclose(s_int, s, atol=1e-6 * max(D, 1e-3))
    # (the trapezoidal profile's acceleration jumps; integrating a step with the
    #  trapezoidal rule is off by up to a*dt/2 per jump, so the tolerance scales with dt)
    v_int = np.concatenate([[0.0], np.cumsum(0.5 * (sdd[1:] + sdd[:-1]) * np.diff(t))])
    assert np.allclose(v_int, sd, atol=2 * a * (t[1] - t[0]) + 1e-12)


@pytest.mark.parametrize('D,v,a,j', CASES)
def test_trapezoidal(D, v, a, j):
    _check(TrapezoidalProfile(D, v, a), D, v, a)


@pytest.mark.parametrize('D,v,a,j', CASES)
def test_scurve(D, v, a, j):
    prof = SCurveProfile(D, v, a, j)
    _check(prof, D, v, a, j)
    # S-curve velocity is continuous AND smooth: acceleration continuous
    t = np.linspace(0.0, prof.duration, 20001)
    _, _, sdd, _ = prof.sample(t)
    assert np.max(np.abs(np.diff(sdd))) < 1e-3 * a + j * (t[1] - t[0]) * 1.01


def test_trapezoidal_known_values():
    p = TrapezoidalProfile(1.0, 0.5, 1.0)         # T_a = 0.5, T_v = 1.5, T = 2.5
    assert p.T_a == pytest.approx(0.5) and p.duration == pytest.approx(2.5)
    s, sd, _, _ = p.sample(np.array([0.5, 1.25]))
    assert s[0] == pytest.approx(0.125) and sd[1] == pytest.approx(0.5)


def test_scurve_slower_than_trapezoidal_but_smoother():
    tp, sc = TrapezoidalProfile(0.5, 0.25, 0.5), SCurveProfile(0.5, 0.25, 0.5, 2.5)
    assert sc.duration > tp.duration                # limited jerk costs a little time
    assert sc.duration - tp.duration == pytest.approx(0.5 / 2.5, rel=1e-6)   # = a/j when v and a are reached


def test_zero_distance_and_factory():
    for prof in (TrapezoidalProfile(0.0, 1, 1), SCurveProfile(0.0, 1, 1, 1)):
        assert prof.duration == 0.0
        s, sd, sdd, _ = prof.sample(np.array([0.0]))
        assert s[0] == sd[0] == sdd[0] == 0.0
    assert isinstance(make_profile('scurve', 1, 1, 1, 1), SCurveProfile)
    with pytest.raises(ValueError):
        make_profile('bang-bang', 1, 1, 1)
