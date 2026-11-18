"""Plot a recorded pick & place run: planned vs measured joint velocities, and Box B.

    ros2 launch gp7_control pick_place.launch.py log_dir:=runs/run1
    ros2 run gp7_motion plot_run runs/run1     # writes tracking.png, tcp_tracking.png, box_b.png

Reads the CSV logs of pick_place_node (planned.csv, measured_joints.csv, box_b.csv,
steps.csv). Each planned arm segment is placed at the time that segment really started
(steps.csv), so planned and measured curves line up even though gripper steps and
action round trips take slightly longer than planned.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gp7_kinematics import JOINT_NAMES, GP7Kinematics  # noqa: E402

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID, SHADE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#eeeeea'


def _read(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(MUTED)


def load(log_dir):
    hp, planned = _read(os.path.join(log_dir, 'planned.csv'))
    _, meas = _read(os.path.join(log_dir, 'measured_joints.csv'))
    _, box = _read(os.path.join(log_dir, 'box_b.csv'))
    _, steps = _read(os.path.join(log_dir, 'steps.csv'))
    seg = {}
    for r in planned:
        seg.setdefault(r[1], []).append([float(x) for x in [r[0]] + r[3:]])
    seg = {k: np.array(v) for k, v in seg.items()}
    for k, v in seg.items():          # time from the segment's own start
        v[:, 0] -= v[0, 0]
    return seg, np.array(meas, float), np.array(box, float), steps


def plot_tracking(seg, meas, steps, out):
    fig, axs = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True)
    t_m = meas[:, 0]
    starts = {s[0]: float(s[2]) for s in steps if s[1] == 'arm'}
    carry = [(float(s[2]), float(s[3])) for s in steps if s[0] in ('lift', 'transfer', 'descend')]
    err = []
    for j in range(6):
        ax = axs.flat[j]
        for a, b in carry:
            ax.axvspan(a, b, color=SHADE, linewidth=0, zorder=0)
        ax.plot(t_m, np.degrees(meas[:, 7 + j]), color=ORANGE, linewidth=1.4, label='measured')
        first = True
        for name, t0 in starts.items():
            if name not in seg:
                continue
            p = seg[name]
            ax.plot(p[:, 0] + t0, np.degrees(p[:, 7 + j]), color=BLUE, linewidth=1.1,
                    linestyle='--', label='planned' if first else None)
            first = False
            # tracking error: measured velocity interpolated at the planned sample times
            qd_m = np.interp(p[:, 0] + t0, t_m, meas[:, 7 + j])
            err.append(np.abs(qd_m - p[:, 7 + j]).max())
        ax.set_title(JOINT_NAMES[j], color=INK, fontsize=10)
        if j % 3 == 0:
            ax.set_ylabel('velocity [deg/s]', color=INK2, fontsize=9)
        if j >= 3:
            ax.set_xlabel('time [s]', color=INK2, fontsize=9)
        _style(ax)
    axs.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Joint velocity: planned profile (dashed) vs measured in Gazebo '
                 f'(shaded: carrying Box B; max deviation {np.degrees(max(err)):.1f} deg/s)',
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'tracking.png'), dpi=110)
    plt.close(fig)
    return np.degrees(max(err))


def _angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _distance_to_polyline(pts, line):
    """Distance from each point to the polyline through `line` (both (N,3))."""
    a, b = line[:-1], line[1:]
    ab = b - a
    L2 = np.maximum((ab * ab).sum(1), 1e-18)
    d = np.empty(len(pts))
    for i, p in enumerate(pts):
        u = np.clip(((p - a) * ab).sum(1) / L2, 0.0, 1.0)
        d[i] = np.sqrt(((a + u[:, None] * ab - p) ** 2).sum(1)).min()
    return d


CARTESIAN = ('lower', 'approach', 'lift', 'transfer', 'descend', 'retreat')


def plot_tcp(seg, meas, steps, out):
    """Offline plan vs online execution in CARTESIAN space.

    For every planned sample, the measured joints at that moment are put through the
    forward kinematics and compared with the planned TCP pose:
      - error vs the planned pose AT THE SAME TIME: includes any timing lag (being a few
        ms behind on the path shows up as speed x lag);
      - distance to the planned PATH (Cartesian segments only): purely geometric, i.e.
        did the gripper stay on the straight line / arc, regardless of timing;
      - orientation error (angle between the planned and the actual gripper orientation).
    """
    kin = GP7Kinematics.from_config(tool='tcp')
    starts = {s[0]: float(s[2]) for s in steps if s[1] == 'arm'}
    carry = [(float(s[2]), float(s[3])) for s in steps if s[0] in ('lift', 'transfer', 'descend')]
    t_m = meas[:, 0]
    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0])
    ax_p, ax_r = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_xy = fig.add_subplot(gs[:, 1])
    worst = {'pos': 0.0, 'rot': 0.0, 'pos_carry': 0.0, 'rot_carry': 0.0, 'path': 0.0}
    first = True
    first_path = True
    for name, t0 in starts.items():
        if name not in seg:
            continue
        p = seg[name]
        t_abs = p[:, 0] + t0
        q_plan = p[:, 1:7]
        q_meas = np.column_stack([np.interp(t_abs, t_m, meas[:, 1 + j]) for j in range(6)])
        T_p = [kin.fk(q) for q in q_plan]
        T_m = [kin.fk(q) for q in q_meas]
        dp = np.array([np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(T_p, T_m)]) * 1000.0
        dr = np.array([_angle_deg(a[:3, :3], b[:3, :3]) for a, b in zip(T_p, T_m)])
        ax_p.plot(t_abs, dp, color=BLUE, linewidth=1.2, label='vs planned pose at the same time' if first else None)
        ax_r.plot(t_abs, dr, color=BLUE, linewidth=1.2)
        pp = np.array([a[:3, 3] for a in T_p])
        pm = np.array([b[:3, 3] for b in T_m])
        if name in CARTESIAN and len(pp) > 1:
            dpath = _distance_to_polyline(pm, pp) * 1000.0
            ax_p.plot(t_abs, dpath, color=ORANGE, linewidth=1.2,
                      label='distance to the planned path (Cartesian segments)' if first_path else None)
            worst['path'] = max(worst['path'], dpath.max())
            first_path = False
        ax_xy.plot(pp[:, 0], pp[:, 1], color=BLUE, linewidth=2.2, label='planned (offline)' if first else None)
        ax_xy.plot(pm[:, 0], pm[:, 1], color=ORANGE, linewidth=1.0, linestyle='--',
                   label='measured (Gazebo)' if first else None)
        first = False
        worst['pos'], worst['rot'] = max(worst['pos'], dp.max()), max(worst['rot'], dr.max())
        if name in ('lift', 'transfer', 'descend'):
            worst['pos_carry'] = max(worst['pos_carry'], dp.max())
            worst['rot_carry'] = max(worst['rot_carry'], dr.max())
    for ax in (ax_p, ax_r):
        for a, b in carry:
            ax.axvspan(a, b, color=SHADE, linewidth=0, zorder=0)
        _style(ax)
    ax_p.set_ylabel('TCP position error [mm]', color=INK2, fontsize=9)
    ax_r.set_ylabel('TCP orientation error [deg]', color=INK2, fontsize=9)
    ax_r.set_xlabel('time [s]', color=INK2, fontsize=9)
    ax_p.set_title(f'vs same-time pose: max {worst["pos"]:.2f} mm ({worst["pos_carry"]:.2f} mm carrying, shaded); '
                   f'off the planned path: max {worst["path"]:.3f} mm', color=INK, fontsize=10)
    ax_p.legend(frameon=False, fontsize=8, loc='upper center')
    ax_r.set_title(f'max {worst["rot"]:.3f} deg overall, {worst["rot_carry"]:.3f} deg while carrying',
                   color=INK, fontsize=10)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlabel('x [m]', color=INK2)
    ax_xy.set_ylabel('y [m]', color=INK2)
    ax_xy.set_title('TCP path, top view', color=INK, fontsize=10)
    ax_xy.legend(frameon=False, fontsize=9)
    _style(ax_xy)
    fig.suptitle('Offline plan vs online execution: gripper TCP pose '
                 '(forward kinematics of the measured joints vs the planned joints)', color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'tcp_tracking.png'), dpi=110)
    plt.close(fig)
    return worst


def plot_box(box, steps, out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t = box[:, 0]
    for ax in (ax1, ax2):
        for s in steps:
            if s[0] in ('lift', 'transfer', 'descend'):
                ax.axvspan(float(s[2]), float(s[3]), color=SHADE, linewidth=0, zorder=0)
            ax.axvline(float(s[2]), color=GRID, linewidth=0.8)
    ax1.plot(t, box[:, 3], color=BLUE, linewidth=1.5)
    ax1.set_ylabel('Box B height [m]', color=INK2, fontsize=9)
    ax2.plot(t, box[:, 4], color=BLUE, linewidth=1.5)
    ax2.set_ylabel('Box B tilt [deg]', color=INK2, fontsize=9)
    ax2.set_ylim(-0.05, max(0.5, box[:, 4].max() * 1.2))
    ax2.set_xlabel('time [s]', color=INK2)
    for ax in (ax1, ax2):
        _style(ax)
    for s in steps:
        ax1.annotate(s[0], (float(s[2]), ax1.get_ylim()[1]), rotation=90, fontsize=7, color=INK2,
                     ha='right', va='top')
    carried = box[box[:, 5] > 0.5, 4]
    fig.suptitle(f'Box B in Gazebo (shaded: carried) - max tilt while carried '
                 f'{carried.max() if len(carried) else float("nan"):.3f} deg', color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'box_b.png'), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log_dir')
    ap.add_argument('--out', help='output directory (default: log_dir)')
    a = ap.parse_args()
    out = a.out or a.log_dir
    os.makedirs(out, exist_ok=True)
    seg, meas, box, steps = load(a.log_dir)
    dev = plot_tracking(seg, meas, steps, out)
    w = plot_tcp(seg, meas, steps, out)
    plot_box(box, steps, out)
    carried = box[box[:, 5] > 0.5, 4]
    print(f'max joint velocity deviation (measured - planned): {dev:.2f} deg/s')
    print(f'max TCP deviation vs same-time planned pose: {w["pos"]:.2f} mm / {w["rot"]:.3f} deg overall, '
          f'{w["pos_carry"]:.2f} mm / {w["rot_carry"]:.3f} deg while carrying')
    print(f'max distance from the planned Cartesian path: {w["path"]:.3f} mm')
    print(f'Box B max tilt while carried: {carried.max() if len(carried) else float("nan"):.3f} deg')
    print(f'figures written to {os.path.abspath(out)}/ (tracking.png, tcp_tracking.png, box_b.png)')


if __name__ == '__main__':
    main()
