"""Plot a recorded pick & place run: planned vs measured joint velocities, and Box B.

    ros2 launch gp7_control pick_place.launch.py log_dir:=runs/run1
    ros2 run gp7_motion plot_run runs/run1            # writes runs/run1/tracking.png, box_b.png

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

from gp7_kinematics import JOINT_NAMES  # noqa: E402

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
    plot_box(box, steps, out)
    print(f'max joint velocity deviation (measured - planned): {dev:.2f} deg/s')
    print(f'figures written to {os.path.abspath(out)}/ (tracking.png, box_b.png)')


if __name__ == '__main__':
    main()
