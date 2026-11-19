"""Plan the pick & place offline and plot it (figures for the report).

    ros2 run gp7_motion plot_plan 0.55 -0.30                 # Box B at (x, y), yaw 0
    ros2 run gp7_motion plot_plan 0.55 -0.30 --yaw 25 --out plots
    ros2 run gp7_motion plot_plan -0.2 0.6 --profile trapezoidal

Writes four PNGs to --out (default ./plan_plots):
  profiles.png     trapezoidal vs S-curve time scaling for the transfer (s, s_dot, s_ddot, jerk)
  joints.png       q, q_dot, q_ddot of every joint over the whole task, with the limits
  path.png         TCP path in 3D and from above, free moves vs carrying Box B
  orientation.png  tilt of the gripper axis from vertical over time (0 while carrying)
and prints a per-segment summary.
"""
import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gp7_kinematics import JOINT_NAMES  # noqa: E402

from .profiles import SCurveProfile, TrapezoidalProfile  # noqa: E402
from .sequence import load_config, make_planner, plan_pick_place  # noqa: E402

# Validated categorical palette (slots 1-3) + neutral ink / grid colours
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID, SHADE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#eeeeea'


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(MUTED)


def default_config():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('gp7_motion'), 'config', 'pick_place.yaml')
    except Exception:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'config', 'pick_place.yaml')


def timeline(plan):
    """Concatenate all steps (gripper steps = robot holds still) into one time axis."""
    t, q, qd, qdd, tcp, tilt, carry, marks = [], [], [], [], [], [], [], []
    t0, q_last = 0.0, None
    for st in plan.steps:
        if st.kind == 'arm':
            tr = st.trajectory
            z_axis = np.array([plan_fk(tr, k) for k in range(len(tr.t))])
            t.append(tr.t + t0); q.append(tr.q); qd.append(tr.qd); qdd.append(tr.qdd); tcp.append(tr.tcp)
            tilt.append(np.degrees(np.arccos(np.clip(-z_axis[:, 2], -1, 1))))
            carry.append(np.full(len(tr.t), st.carrying))
            q_last = tr.q_end
        else:
            if q_last is None:
                marks.append((t0, t0 + st.duration, st.name))
                t0 += st.duration
                continue
            tt = np.array([t0, t0 + st.duration])
            t.append(tt); q.append(np.vstack([q_last, q_last])); qd.append(np.zeros((2, 6)))
            qdd.append(np.zeros((2, 6))); tcp.append(np.vstack([tcp[-1][-1]] * 2))
            tilt.append(np.full(2, tilt[-1][-1])); carry.append(np.full(2, st.carrying))
        marks.append((t0, t0 + st.duration, st.name))
        t0 += st.duration
    return (np.concatenate(t), np.vstack(q), np.vstack(qd), np.vstack(qdd), np.vstack(tcp),
            np.concatenate(tilt), np.concatenate(carry), marks)


_KIN = {}


def plan_fk(tr, k):
    return _KIN['kin'].fk(tr.q[k])[:3, 2]


def shade_carry(ax, t, carry):
    """Light band where Box B is carried."""
    on = False
    for i in range(len(t)):
        if carry[i] and not on:
            start, on = t[i], True
        if on and (not carry[i] or i == len(t) - 1):
            ax.axvspan(start, t[i], color=SHADE, zorder=0, linewidth=0)
            on = False


def plot_profiles(cfg, plan, out):
    tr = next(s.trajectory for s in plan.steps if s.name == 'transfer')
    lim = cfg['cartesian']['transfer']
    D = tr.path.length
    profs = [('trapezoidal', TrapezoidalProfile(D, lim['v'], lim['a']), BLUE),
             ('S-curve', SCurveProfile(D, lim['v'], lim['a'], lim['j']), ORANGE)]
    fig, axs = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    labels = ['s  [m]', 'ṡ  [m/s]', 's̈  [m/s²]', 'jerk  [m/s³]']
    for name, prof, col in profs:
        t = np.linspace(0, prof.duration, 3000)
        vals = prof.sample(t)
        for ax, y in zip(axs, vals):
            ax.plot(t, y, color=col, linewidth=1.6, label=f'{name} (T = {prof.duration:.2f} s)')
    for ax, lab in zip(axs, labels):
        ax.set_ylabel(lab, color=INK2, fontsize=9)
        _style(ax)
    axs[1].axhline(lim['v'], color=MUTED, linestyle='--', linewidth=0.8)
    for y in (lim['a'], -lim['a']):
        axs[2].axhline(y, color=MUTED, linestyle='--', linewidth=0.8)
    for y in (lim['j'], -lim['j']):
        axs[3].axhline(y, color=MUTED, linestyle='--', linewidth=0.8)
    axs[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    axs[-1].set_xlabel('time [s]', color=INK2)
    fig.suptitle(f'Transfer segment time scaling (path length {D:.3f} m, '
                 f'v ≤ {lim["v"]} m/s, a ≤ {lim["a"]} m/s², j ≤ {lim["j"]} m/s³)', color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'profiles.png'), dpi=130)
    plt.close(fig)


def plot_joints(planner, plan, out):
    t, q, qd, qdd, _, _, carry, marks = timeline(plan)
    fig, axs = plt.subplots(3, 6, figsize=(18, 8.5), sharex=True)
    rows = [(np.degrees(q), 'position [deg]', None),
            (np.degrees(qd), 'velocity [deg/s]', np.degrees(planner.lim.v)),
            (np.degrees(qdd), 'acceleration [deg/s²]', np.degrees(planner.lim.a))]
    for r, (data, lab, lim) in enumerate(rows):
        for j in range(6):
            ax = axs[r, j]
            shade_carry(ax, t, carry)
            ax.plot(t, data[:, j], color=BLUE, linewidth=1.3)
            if lim is not None:
                for y in (lim[j], -lim[j]):
                    ax.axhline(y, color=MUTED, linestyle='--', linewidth=0.8)
            else:
                ax.axhline(np.degrees(planner.lim.lower[j]), color=MUTED, linestyle='--', linewidth=0.8)
                ax.axhline(np.degrees(planner.lim.upper[j]), color=MUTED, linestyle='--', linewidth=0.8)
                ax.set_ylim(min(data[:, j].min(), 0) - 30, max(data[:, j].max(), 0) + 30)
            _style(ax)
            if r == 0:
                ax.set_title(JOINT_NAMES[j], color=INK, fontsize=10)
            if j == 0:
                ax.set_ylabel(lab, color=INK2, fontsize=9)
            if r == 2:
                ax.set_xlabel('time [s]', color=INK2, fontsize=9)
    fig.suptitle('Joint trajectories of the whole pick & place  '
                 '(dashed: limits used by the planner; shaded: carrying Box B)', color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'joints.png'), dpi=110)
    plt.close(fig)


def _box_edges(cx, cy, size, h, z0=0.0):
    s = size / 2
    xs = [cx - s, cx + s, cx + s, cx - s, cx - s]
    ys = [cy - s, cy - s, cy + s, cy + s, cy - s]
    return xs, ys, z0, z0 + h


def plot_path(cfg, plan, box_b, out):
    fig = plt.figure(figsize=(14, 6.5))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    seen = set()
    for st in plan.arm_steps:
        tr = st.trajectory
        col, lab = (BLUE, 'carrying Box B (Cartesian, constant orientation)') if st.carrying else \
                   (ORANGE, 'free move (joint space)') if tr.kind == 'joint' else \
                   (AQUA, 'approach / retreat (Cartesian)')
        kw = {'label': lab} if lab not in seen else {}
        seen.add(lab)
        ax3.plot(*tr.tcp.T, color=col, linewidth=1.8, **kw)
        ax2.plot(tr.tcp[:, 0], tr.tcp[:, 1], color=col, linewidth=1.8, **kw)
    a = cfg['task']['box_a']
    for (cx, cy, size, h, name) in [(a['x'], a['y'], 0.30, a['height'], 'Box A'),
                                    (box_b[0], box_b[1], cfg['task']['box_b_size'], cfg['task']['box_b_size'], 'Box B')]:
        xs, ys, z0, z1 = _box_edges(cx, cy, size, h)
        for z in (z0, z1):
            ax3.plot(xs, ys, [z] * 5, color=MUTED, linewidth=0.9)
        ax2.plot(xs, ys, color=MUTED, linewidth=0.9)
        ax2.annotate(name, (cx, cy), color=INK2, fontsize=9, ha='center', va='center')
    ax2.add_patch(plt.Circle((0, 0), 0.1, fill=False, color=MUTED, linewidth=0.9))
    ax2.annotate('robot base', (0, 0), color=INK2, fontsize=8, ha='center', va='center')
    ax2.set_aspect('equal')
    _style(ax2)
    ax2.set_xlabel('x [m]', color=INK2)
    ax2.set_ylabel('y [m]', color=INK2)
    ax2.set_title('top view', color=INK, fontsize=10)
    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('y [m]')
    ax3.set_zlabel('z [m]')
    ax3.set_title('TCP path', color=INK, fontsize=10)
    ax2.legend(frameon=False, fontsize=9, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'path.png'), dpi=120)
    plt.close(fig)


def plot_orientation(plan, out):
    t, _, _, _, _, tilt, carry, marks = timeline(plan)
    fig, ax = plt.subplots(figsize=(11, 4))
    shade_carry(ax, t, carry)
    ax.plot(t, tilt, color=BLUE, linewidth=1.4)
    for (t0, t1, name) in marks:
        ax.axvline(t0, color=GRID, linewidth=0.8)
        ax.annotate(name, ((t0 + t1) / 2, ax.get_ylim()[1] if tilt.max() > 0 else 1), rotation=90,
                    fontsize=7, color=INK2, ha='center', va='top')
    _style(ax)
    ax.set_xlabel('time [s]', color=INK2)
    ax.set_ylabel('gripper axis tilt from vertical [deg]', color=INK2, fontsize=9)
    carry_max = tilt[carry].max() if carry.any() else 0.0
    ax.set_title(f'Box B orientation: max tilt while carrying (shaded) = {carry_max:.2e} deg',
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'orientation.png'), dpi=120)
    plt.close(fig)


def summary(planner, plan):
    print(f'Grasp/carry yaw psi = {math.degrees(plan.psi):.1f} deg; total time {plan.duration:.2f} s')
    print(f'{"step":13s} {"kind":10s} {"T [s]":>6s} {"max|qd|/v":>9s} {"max|qdd|/a":>10s} '
          f'{"stretch":>7s} {"orient. err [deg]":>17s}')
    for st in plan.steps:
        if st.kind == 'gripper':
            print(f'{st.name:13s} {"gripper":10s} {st.duration:6.2f}   pad gap -> {st.width * 1000:.0f} mm '
                  f'(knuckle {st.theta:.3f} rad)')
            continue
        tr = st.trajectory
        oe = f'{math.degrees(np.max(tr.orientation_error)):.1e}' if tr.kind == 'cartesian' else '(free)'
        print(f'{tr.name:13s} {tr.kind:10s} {tr.duration:6.2f} {np.max(np.abs(tr.qd) / planner.lim.v):9.2f} '
              f'{np.max(np.abs(tr.qdd) / planner.lim.a):10.2f} {tr.stretch:7.2f} {oe:>17s}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('x', type=float, help='Box B centre x [m]')
    ap.add_argument('y', type=float, help='Box B centre y [m]')
    ap.add_argument('--yaw', type=float, default=0.0, help='Box B yaw [deg]')
    ap.add_argument('--box-a', nargs=2, type=float, metavar=('X', 'Y'), help='override Box A centre')
    ap.add_argument('--profile', choices=['scurve', 'trapezoidal'], help='override the config profile')
    ap.add_argument('--config', default=default_config())
    ap.add_argument('--out', default='plan_plots')
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.box_a:
        cfg['task']['box_a']['x'], cfg['task']['box_a']['y'] = a.box_a
    planner = make_planner(cfg)
    _KIN['kin'] = planner.kin
    box_b = (a.x, a.y, 0.0, math.radians(a.yaw))
    plan = plan_pick_place(planner, cfg, box_b, np.zeros(6), profile=a.profile)
    summary(planner, plan)
    os.makedirs(a.out, exist_ok=True)
    plot_profiles(cfg, plan, a.out)
    plot_joints(planner, plan, a.out)
    plot_path(cfg, plan, box_b, a.out)
    plot_orientation(plan, a.out)
    print(f'figures written to {os.path.abspath(a.out)}/')


if __name__ == '__main__':
    main()
