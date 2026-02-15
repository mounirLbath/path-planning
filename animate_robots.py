"""
This script has been written by ChatGPT to provide a quick way to animate the paths found by the PSO_robots.py script.

Animate two robots moving along their planned paths.

- Robots move at constant speed (default: 1 m/s).
- Each robot has a safety zone: disk of radius R = problem.safety_radius.
- After reaching its goal, a robot waits at its goal (so late collisions are visible).

Usage (example):
  python animate_robots.py 0

This script will:
  - load scenario{n}.txt
  - run PSO_robots.particle_swarm_optimization(...) to get two paths
  - animate the resulting motion with safety zones
"""

from __future__ import annotations

import time
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from environment import Problem, Point, Path, load_problem


def _polyline_points(path: Path) -> List[Point]:
    return [path.start, *path.points, path.end]


def _dist(a: Point, b: Point) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


def _point_at_distance(path: Path, d: float) -> Point:
    """
    Return point at curvilinear distance d along polyline path.
    d<=0 -> start, d>=length -> end.
    """
    pts = _polyline_points(path)
    if not pts:
        return Point(0, 0)
    if d <= 0:
        return pts[0]

    remaining = d
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        L = _dist(a, b)
        if L <= 1e-12:
            continue
        if remaining <= L:
            t = remaining / L
            return a * (1 - t) + b * t
        remaining -= L
    return pts[-1]


def _pos_at_time(path: Path, t: float, speed: float) -> Point:
    """Robot position at time t with constant speed; wait at goal after arrival."""
    if speed <= 0:
        return path.start
    return _point_at_distance(path, max(0.0, t) * speed)


def animate_paths(
    problem: Problem,
    path1: Path,
    path2: Path,
    speed: float = 1.0,
    playback_seconds: float = 8.0,
    fps: int = 30,
    trail: bool = True,
) -> None:
    """
    Show an interactive animation.

    - speed: m/s (assignment says 1)
    - playback_seconds: total animation duration (time-compressed playback)
    - fps: frames per second
    - trail: if True, show traveled trajectory
    """
    R = problem.safety_radius
    T1 = path1.length() / speed if speed > 0 else 0.0
    T2 = path2.length() / speed if speed > 0 else 0.0
    Tmax = max(T1, T2)
    fps = max(int(fps), 1)
    playback_seconds = max(float(playback_seconds), 0.5)
    n_frames = max(int(playback_seconds * fps), 2)
    dt_sim = (Tmax / (n_frames - 1)) if Tmax > 0 else 0.0
    interval_ms = max(1, int(1000 / fps))

    fig, ax = plt.subplots()
    ax.set_xlim(0, problem.xmax)
    ax.set_ylim(0, problem.ymax)
    ax.set_aspect("equal")

    # Obstacles
    for obs in problem.obstacles:
        ax.add_patch(plt.Rectangle((obs.x, obs.y), obs.width, obs.height, color="black"))

    # Planned paths (static)
    ax.plot(
        [path1.start.x, *[p.x for p in path1.points], path1.end.x],
        [path1.start.y, *[p.y for p in path1.points], path1.end.y],
        "r--",
        linewidth=1.0,
        label="path robot 1",
        alpha=0.6,
    )
    ax.plot(
        [path2.start.x, *[p.x for p in path2.points], path2.end.x],
        [path2.start.y, *[p.y for p in path2.points], path2.end.y],
        "g--",
        linewidth=1.0,
        label="path robot 2",
        alpha=0.6,
    )

    # Start/goal markers
    ax.plot(problem.start1.x, problem.start1.y, "ro", markersize=7, label="start1")
    ax.plot(problem.goal1.x, problem.goal1.y, "r*", markersize=10, label="goal1")
    ax.plot(problem.start2.x, problem.start2.y, "go", markersize=7, label="start2")
    ax.plot(problem.goal2.x, problem.goal2.y, "g*", markersize=10, label="goal2")

    # Robots (dynamic)
    robot1_dot, = ax.plot([], [], "ro", markersize=8)
    robot2_dot, = ax.plot([], [], "go", markersize=8)
    safe1 = plt.Circle((problem.start1.x, problem.start1.y), R, color="red", alpha=0.15)
    safe2 = plt.Circle((problem.start2.x, problem.start2.y), R, color="green", alpha=0.15)
    ax.add_patch(safe1)
    ax.add_patch(safe2)

    # Trails
    trail1_line, = ax.plot([], [], "r-", linewidth=1.5, alpha=0.9)
    trail2_line, = ax.plot([], [], "g-", linewidth=1.5, alpha=0.9)
    trail1_x: List[float] = []
    trail1_y: List[float] = []
    trail2_x: List[float] = []
    trail2_y: List[float] = []

    title = ax.text(0.02, 1.02, "", transform=ax.transAxes)
    collision_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    def init():
        robot1_dot.set_data([], [])
        robot2_dot.set_data([], [])
        trail1_line.set_data([], [])
        trail2_line.set_data([], [])
        title.set_text("")
        collision_text.set_text("")
        return robot1_dot, robot2_dot, safe1, safe2, trail1_line, trail2_line, title, collision_text

    def update(frame: int):
        t = min(frame * dt_sim, Tmax)
        p1 = _pos_at_time(path1, t, speed)
        p2 = _pos_at_time(path2, t, speed)

        robot1_dot.set_data([p1.x], [p1.y])
        robot2_dot.set_data([p2.x], [p2.y])

        safe1.center = (p1.x, p1.y)
        safe2.center = (p2.x, p2.y)

        if trail:
            trail1_x.append(p1.x)
            trail1_y.append(p1.y)
            trail2_x.append(p2.x)
            trail2_y.append(p2.y)
            trail1_line.set_data(trail1_x, trail1_y)
            trail2_line.set_data(trail2_x, trail2_y)

        d = _dist(p1, p2)
        title.set_text(
            f"t = {t:.2f}s (speed={speed:.1f} m/s)  |  playback ~{playback_seconds:.1f}s @ {fps} fps"
        )
        if d < 2 * R:
            collision_text.set_text(f"WARNING: safety zones overlap (d={d:.2f} < {2*R:.2f})")
            collision_text.set_color("crimson")
        else:
            collision_text.set_text(f"d(robot1, robot2) = {d:.2f}")
            collision_text.set_color("black")

        return robot1_dot, robot2_dot, safe1, safe2, trail1_line, trail2_line, title, collision_text

    # IMPORTANT: keep a reference to the animation object,
    # otherwise it may get garbage-collected before rendering.
    anim_ref = {"anim": None}

    def _make_anim():
        return FuncAnimation(
            fig,
            update,
            frames=n_frames,
            init_func=init,
            interval=interval_ms,  # ms
            blit=True,
            repeat=False,
        )

    anim_ref["anim"] = _make_anim()
    # Extra safeguard (common Matplotlib pattern)
    fig._anim = anim_ref["anim"]  # type: ignore[attr-defined]

    # Replay button (restarts animation without rerunning PSO)
    btn_ax = fig.add_axes([0.40, 0.01, 0.20, 0.06])  # [left, bottom, width, height]
    replay_btn = Button(btn_ax, "Replay")

    def _on_replay(_event=None):
        if anim_ref["anim"] is not None:
            try:
                anim_ref["anim"].event_source.stop()
            except Exception:
                pass

        # Clear trails
        trail1_x.clear()
        trail1_y.clear()
        trail2_x.clear()
        trail2_y.clear()

        # Reset artists
        init()

        # Recreate and start animation
        anim_ref["anim"] = _make_anim()
        fig._anim = anim_ref["anim"]  # type: ignore[attr-defined]
        try:
            anim_ref["anim"].event_source.start()
        except Exception:
            pass
        fig.canvas.draw_idle()

    replay_btn.on_clicked(_on_replay)

    # Leave room for the button at the bottom
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()
