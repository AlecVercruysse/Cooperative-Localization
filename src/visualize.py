import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import numpy as np
import pandas as pd
# from tqdm import tqdm

# import pdb
import code

colors = ["red", "orange", "lime", "cyan", "orchid"]


class RobotVisual:
    """
    This class controls the plotting of a single robot instance.
    It can be used to perform animations. A robot consists of a circle,
    a line describing the orientation of the robot, and a text annotation with
    the robot's ID.
    """

    def __init__(self, ax, x, y, theta, name="", color="red"):
        """
        Create a robot on an axis.

        Parameters:
        ----------
        ax : matplotlib axes
           matplotlib axes on which to plot the robot.
        x : float
           x location of center of robot
        y : float
           y location of center of robot
        theta : float
           angle of orientation of robot (in radians)
        name : string
           string (e.g. ID) of robot
        color : string
           color to pass to matplotlib as face color of circle.

        Returns:
        -------
        None
        """
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
        """
        Add the circle and line to the axis.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """
        self.ax.add_patch(self.circle)
        self.ax.add_line(self.line)

    def update(self, x, y, theta):
        """
        Update the position of the robot.

        Parameters:
        ----------
        x : float
           x location of center of robot
        y : float
           y location of center of robot
        theta : float
           angle of orientation of robot (in radians)

        Returns:
        -------
        None
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.circle.set_center((self.x, self.y))
        self.line.set_data([x, x+self.wheelbase/2*np.cos(self.theta)],
                           [y, y+self.wheelbase/2*np.sin(self.theta)])
        self.label.set_position((x+self.wheelbase/2, y+self.wheelbase/2))


class SceneAnimation:
    """
    This class construct an animation consisting of scene data:
     - ground-truth landmarks
     - robot positions
     - todo: robot measurements of landmarks?

    This class uses FuncAnimation from matplotlib, and includes a method
    to save the animation to a file.
    """

    def __init__(self, dfs, landmark_gt, title="",
                 speedup=20, fs=50, undersample=100,
                 figsize=(5, 8),
                 keys=["gt_x", "gt_y", "gt_theta"]):
        """
        Construct the animation.

        Parameters:
        ----------
        dfs: list of pandas.DataFrame
           A dataframe for each robot to plot, containing x, y, and orientation
           data. A crucial assumption is that the length (time) of each
           dataframe is constant. This function is written to work with the
           output of file_tools.get_dataset() by default.
        landmark_gt: pandas.DataFrame
           A dataframe containing landmark ground truth information, generated
           by file_tools.get_dataset().
        title: str, optional
           The title of the animation.
        speedup: float, optional
           How much faster to run the animation than real-time.
        fs: float, optional
           The sampling rate of the dataset. Defaults to the default sampling
           rate of file_tools.get_dataset()
        undersample: integer, optional
           How many frames to skip when creating the gif. e.g. undersample=10
           means only plot every 10th frame. This is what to use to try and
           speed up the GIF creation time.
        figsize: tuple, optional
           (width, height) in inches.
        keys: list, length 3, of strings, optional
            The keys to use to access the robot dataframe to extract location
            info. This defaults to the robot ground-truth keys, but in case
            estimates are to be plotted instead, the keys can be set to
            anything. Order is (x, y, orientation).

        Returns:
        -------
        self: SceneAnimation object
        """
        self.dfs = dfs
        self.landmark_gt = landmark_gt
        self.keys = keys
        self.title = title
        self.fs = fs
        self.length = len(self.dfs[0])  # in frames, of the actual dataset
        self.figsize = figsize

        self.xb, self.yb = get_lims(dfs, landmark_gt)
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.interval = int(1000*undersample/fs/speedup)
        
        pause_time = 3  # seconds
        pause_frames = int(pause_time / (1000*undersample/fs/speedup))
        self.ani = FuncAnimation(fig=self.fig,
                                 frames=range(0,
                                              self.length + pause_frames,
                                              undersample),
                                 func=self.update_plot,
                                 init_func=self.start_plot,
                                 interval=self.interval,
                                 blit=False, repeat=False)

    def start_plot(self):
        """
        Called by the constructor. Initialze the animation
        (plot the first frame).
        """
        _, self.robots = plot_one_step([df.iloc[0] for df in self.dfs],
                                       landmark_gt=self.landmark_gt,
                                       x_bounds=self.xb,
                                       y_bounds=self.yb,
                                       ax=self.ax)
        self.ax.set_title(f"{self.title}\n{self._create_time_ctr(0)}")

    def update_plot(self, frame):
        """
        Update function used by Matplotlib's FuncAnimation.
        Called automatically.

        Parameters:
        ----------
        frame : int
           index of the frame to plot. Corresponds to dataset time
           index, but can go over. If it goes over, the last time
           in the dataset is used.
        """
        if frame >= self.length:
            # this must be an ending pause frame
            frame = self.length - 1
        series = [df.iloc[frame] for df in self.dfs]
        for i in range(len(series)):
            x, y, theta = series[i][self.keys]
            self.robots[i].update(x, y, theta)
        self.ax.set_title(f"{self.title}\n{self._create_time_ctr(frame)}")

    def _create_time_ctr(self, frame):
        """
        Internal function to figure out the time of the frame
        to add to the title.
        """
        total_sec = self.length / self.fs
        current_sec = frame / self.fs
        ctr = f"{current_sec:.2f}/{total_sec:.2f} s"
        return ctr

    def write(self, fname):
        """
        Save the animation to a GIF file. This is not called automatically,
        and can take a lot of time.

        Parameters:
        ----------
        fname: str
           File name to write to. This is relative to the directory
           that the script was called in, unless the cwd was changed.
        """
        print(f"writing to {fname}. This can take a while...")
        writer = PillowWriter(fps=1000/self.interval)
        self.ani.save(fname, writer=writer)


def plot_one_step(series, landmark_gt=None,
                  ax=None, x_bounds=None, y_bounds=None, axiseq=True,
                  fname=None, keys=["gt_x", "gt_y", "gt_theta"]):
    """
    Create a matplotlib plot describing a single time frame.

    Parameters:
    ----------
    series: list of dataframes or dataframe
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
    if axiseq:
        ax.axis('equal')
    if fname is not None:
        plt.savefig(fname)

    return ax, robots


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

    # example usage
    dfs, landmark_gt = file_tools.get_dataset(1)
    s = SceneAnimation(dfs, landmark_gt, title="dataset 1")
    print("\n\ncreated s, an animation object. try either" +
          "\ns.write(\"out.gif\") or \nplt.show()\n\n")
    code.interact(local=locals())
    # s.write("out.gif")
    # plt.show()
