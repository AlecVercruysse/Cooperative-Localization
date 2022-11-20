import matplotlib.pyplot as plt
import numpy as np
import imageio  # for creating animations
import pandas as pd
from tqdm import tqdm

import pdb
import tempfile

colors = ["red", "orange", "lime", "cyan", "orchid"]


class RobotVisual:
    def __init__(self, ax, x, y, theta, name="", color="red"):
        self.ax = ax
        self.x = x
        self.y = y
        self.theta = theta
        self.name = name
        self.wheelbase = .235  # m in diameter

        self.circle = plt.Circle((x, y),
                                 radius=self.wheelbase/2,
                                 facecolor=color,
                                 edgecolor="black")
        self.line = plt.Line2D([x, x+self.wheelbase/2*np.cos(self.theta)],
                               [y, y+self.wheelbase/2*np.sin(self.theta)],
                               color="black")
        self.label = self.ax.annotate(name,
                                      (x+self.wheelbase/2, y+self.wheelbase/2),
                                      color="black")

        self.draw()

    def draw(self):
        self.ax.add_patch(self.circle)
        self.ax.add_line(self.line)


def create_animation(dfs, landmark_gt,
                     output="out.gif",
                     speedup=10, fs=50, undersample=100):

    """
    create an animation of the full dataset. TODO: use matplotlib.FuncAnimation
    instead of this jankiness.
    """
    xb, yb = get_lims(dfs, landmark_gt)
    images = []
    print("generating frames...")
    for frame in tqdm(range(0, len(dfs[0]), undersample)):
        fp = tempfile.TemporaryFile()
        plot_one_step([df.iloc[frame] for df in dfs],
                      landmark_gt=landmark_gt,
                      x_bounds=xb,
                      y_bounds=yb,
                      fname=fp)
        fp.seek(0)
        images += [imageio.v2.imread(fp.read())]
    print(f"saving gif output to {output} (this might take a while...)")
    imageio.mimsave(output, images, duration=undersample/fs/speedup)


def plot_one_step(series, landmark_gt=None,
                  ax=None, x_bounds=None, y_bounds=None,
                  fname=None, keys=["gt_x", "gt_y", "gt_theta"]):
    """
    Create a matplotlib plot describing a single time frame.

    Parameters:
    ----------
    series: list of dataframes or datafram
       describing the current time frame.
    """
    if not isinstance(series, list):
        series = [series]
    if not isinstance(series[0], pd.Series):
        raise ValueError("series needs to be a Series" +
                         " or list of Series!")
    if ax is None:
        _, ax = plt.subplots()

    robots = []
    for i in range(len(series)):
        x, y, theta = series[i][keys]
        # print(f"{x=} {y=} {theta=}")
        robots += [RobotVisual(ax, x, y, theta,
                               name=f"{i+1}", color=colors[i])]

    if landmark_gt is not None:
        for i in landmark_gt.index:
            x, y, name = landmark_gt.iloc[i][["x [m]", "y [m]", "Subject #"]]
            plt.scatter(x, y, color="black")
            plt.annotate(name, (x, y))

    # figure out x limits:
    ax.autoscale_view()
    if x_bounds is not None:
        ax.set_xlim(*x_bounds)
    if y_bounds is not None:
        ax.set_ylim(*y_bounds)
    if fname is not None:
        plt.savefig(fname)

    return ax


def get_lims(data, landmark_gt,
             robot_keys=["gt_x", "gt_y", "gt_theta"]):
    """
    get x and y limits to plot with.
    """
    max_robot_x = np.max([np.max(df[robot_keys[0]]) for df in data])
    min_robot_x = np.min([np.min(df[robot_keys[0]]) for df in data])

    max_robot_y = np.max([np.max(df[robot_keys[1]]) for df in data])
    min_robot_y = np.min([np.min(df[robot_keys[1]]) for df in data])

    # now do landmark gt data:
    if landmark_gt is not None:
        min_lm_x = np.min(landmark_gt["x [m]"])
        max_lm_x = np.max(landmark_gt["x [m]"])
        min_lm_y = np.min(landmark_gt["y [m]"])
        max_lm_y = np.max(landmark_gt["y [m]"])

    min_x = min(min_robot_x, min_lm_x)
    max_x = max(max_robot_x, max_lm_x)
    min_y = min(min_robot_y, min_lm_y)
    max_y = max(max_robot_y, max_lm_y)

    r = .235 / 2  # wheelbase / 2

    return (min_x - r/2, max_x + r/2), (min_y - r/2, max_y + r/2)


if __name__ == "__main__":
    import file_tools
    dfs, landmark_gt = file_tools.get_dataset(1)
    create_animation(dfs, landmark_gt)
