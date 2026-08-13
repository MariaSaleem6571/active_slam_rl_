#!/usr/bin/env python3
"""
Live training monitor: opens an actual on-screen matplotlib window and
keeps it updating in real time while training runs in another
terminal/process, by polling the episode-metrics CSV
(`rl/callbacks.py::MetricsLoggingCallback` writes to it every episode).

This is separate from the PNG that training itself keeps overwriting
(`results/training_curves.png`, refreshed automatically every few
episodes -- see MetricsLoggingCallback's `live_plot` option, on by
default) because showing an actual window requires a display and an
interactive matplotlib backend, which isn't something a headless
Docker container or a remote server has. Use this script when you *do*
have a display (running natively, or with X11 forwarding); use the
auto-refreshing PNG everywhere else, including inside Docker.

Usage (run in a separate terminal while training is running):
    python scripts/live_monitor.py --csv results/train_logs/episode_metrics.csv

Close the window, or Ctrl-C in the terminal, to stop watching -- this
never touches training itself, it only reads the CSV.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/train_logs/episode_metrics.csv")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between refreshes")
    args = parser.parse_args()

    print(f"Watching {args.csv} -- waiting for the first episode to be logged...")
    while not os.path.exists(args.csv):
        time.sleep(args.interval)

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.canvas.manager.set_window_title("active-slam-rl -- live training monitor")
    lines = {}  # (ax_idx) -> dict of line handles, created lazily on first draw

    def redraw():
        try:
            df = pd.read_csv(args.csv)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return
        if df.empty:
            return
        window = max(1, len(df) // 40)

        panels = [
            (axes[0, 0], "episode_reward", "tab:blue", "Episode reward"),
            (axes[0, 1], "map_completeness", "tab:green", "Map completeness (%)"),
            (axes[1, 0], "mean_ate", "tab:red", "Mean ATE / drift (m)"),
            (axes[1, 1], "collision_count", "tab:orange", "Collisions / episode"),
        ]
        for ax, col, color, title in panels:
            ax.clear()
            ax.plot(df["timestep"], df[col], alpha=0.3, color=color)
            ax.plot(df["timestep"], df[col].rolling(window, min_periods=1).mean(),
                     color=color, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("training timestep")

        fig.suptitle(f"Live training progress -- {len(df)} episodes, "
                      f"timestep {int(df['timestep'].iloc[-1])}")
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

    try:
        while plt.fignum_exists(fig.number):
            redraw()
            plt.pause(args.interval)
    except KeyboardInterrupt:
        pass
    print("Stopped watching (window closed or interrupted).")


if __name__ == "__main__":
    main()
