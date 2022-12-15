from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np

import pdb
import os
import code

colors = ["red", "orange", "lime", "cyan", "orchid"]


def get_cov_ellipse_params(x, y, cov):
    w, v = np.linalg.eig(cov)
    lambda1, lambda2 = w
    v1, v2 = v.T
    alpha = np.arctan2(v1[1], v1[0])
    major = 2*np.sqrt(5.991*lambda1)
    minor = 2*np.sqrt(5.991*lambda2)
    return alpha, major, minor


def get_cov_ellipse(x, y, cov, color, opacity=1):
    """
    Returns an ellipse patch for a covariance matrix of x,y uncertainty.
    """
    alpha, major, minor = get_cov_ellipse_params(x, y, cov)
    el = mpl.patches.Ellipse((x, y), major, minor, np.rad2deg(alpha),
                             color=(color), alpha=opacity)
    return el


class RobotVisual:
    """
    This class controls the plotting of a single robot instance.
    It can be used to perform animations. A robot consists of a circle,
    a line describing the orientation of the robot, and a text annotation with
    the robot's ID.
    """

    def __init__(self, ax, robot, name="", color="red",
                 plot_est_pos=False,
                 plot_est_landmarks=False,
                 plot_landmark_uncertainty=False,
                 plot_measurements=False,
                 only_robot_measurements=False):
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
        self.robot = robot
        self.plot_est_pos = plot_est_pos
        self.plot_est_landmarks = plot_est_landmarks
        self.plot_landmark_uncertainty = plot_landmark_uncertainty
        self.plot_measurements = plot_measurements
        self.only_robot_measurements = only_robot_measurements
        self.x, self.y, self.theta = self.robot.get_gt(0)
        self.name = name
        self.color = color
        self.wheelbase = .235  # m in diameter

        self.circle = plt.Circle((self.x, self.y),
                                 radius=self.wheelbase/2,
                                 facecolor=color,
                                 edgecolor="black",
                                 label=f"Robot {self.robot.my_idx}")
        self.line = plt.Line2D([self.x,
                                self.x+self.wheelbase/2*np.cos(self.theta)],
                               [self.y,
                                self.y+self.wheelbase/2*np.sin(self.theta)],
                               color="black")
        self.label = self.ax.annotate(name,
                                      (self.x+self.wheelbase/2,
                                       self.y+self.wheelbase/2),
                                      color="black")

        if self.plot_est_pos:
            self.init_est_pose()

        if self.plot_est_landmarks:
            self.init_est_landmarks()

        if self.plot_measurements:
            self.init_measurements()

        self.draw()

    def init_est_pose(self):
        (x, y, theta), cov = self.robot.get_est_pos(0)
        cov = cov[:2, :2]  # just x, y
        self.est_pos = get_cov_ellipse(x, y, cov, color=self.color, opacity=0.1)
        self.est_pose = plt.Line2D([x,
                                    x+self.wheelbase*np.cos(theta)],
                                   [y,
                                    y+self.wheelbase*np.sin(self.theta)],
                                   color="black", alpha=0.7)

    def update_est_pos(self, frame):
        (x, y, theta), cov = self.robot.get_est_pos(frame)
        cov = cov[:2, :2]  # just x, y
        alpha, major, minor = get_cov_ellipse_params(x, y, cov)
        self.est_pos.set_center((x, y))
        self.est_pos.set_angle(np.rad2deg(alpha))
        self.est_pos.set_width(major)
        self.est_pos.set_height(minor)
        self.est_pose.set_data([x, x+self.wheelbase/2*np.cos(theta)],
                               [y, y+self.wheelbase/2*np.sin(theta)])

    def init_est_landmarks(self):
        self.landmarks = np.array(range(5, 20)) + 1
        names, info = zip(*[(idx, self.robot.get_est_landmark(0, idx))
                            for idx in self.landmarks])
        location, cov = zip(*info)
        x, y = np.array(location).T

        if not self.plot_landmark_uncertainty:
            self.landmark_est = self.ax.scatter(x, y, color=self.color)
        else:
            self.landmark_est = [get_cov_ellipse(xi, yi, covi,
                                                 color=self.color, opacity=0.1)
                                 for xi, yi, covi in zip(x, y, cov)]
        self.landmark_labels = []
        for name, xi, yi in zip(names, x, y):
            self.landmark_labels += [plt.annotate(int(name), (xi, yi),
                                                  color=self.color)]

    def update_est_landmarks(self, frame):
        names, info = zip(*[(idx, self.robot.get_est_landmark(frame, idx))
                            for idx in self.landmarks])
        location, cov = zip(*info)
        x, y = np.array(location).T
        if not self.plot_landmark_uncertainty:
            self.landmark_est.set_offsets(np.array([x, y]).T)
        else:
            params = [get_cov_ellipse_params(xi, yi, covi)
                      for xi, yi, covi in zip(x, y, cov)]
            for (alpha, major, minor), lm, xi, yi in zip(params,
                                                         self.landmark_est,
                                                         x, y):
                # pdb.set_trace()
                lm.set_center((xi, yi))
                lm.set_angle(alpha)
                lm.set_width(major)
                lm.set_height(minor)
            
        for i, (xi, yi) in enumerate(zip(x, y)):
            self.landmark_labels[i].set_position((xi, yi))

    def init_measurements(self):
        self.measurement_lines = []
        self.measurement_labels = []
        self.measurement_last_frame = 0

    def update_measurements(self, frame):
        if frame == self.measurement_last_frame:
            return  # we're at the end, don't update
        for line in self.measurement_lines:
            self.ax.lines.remove(line)  # get rid of meas. from last frame
        for annot in self.measurement_labels:
            annot.remove()
        # frames = range(self.measurement_last_frame, frame+1)
        frames = range(frame, frame+10)  # look ahead
        self.measurement_last_frame = frame
        self.measurement_lines = []
        self.measurement_labels = []

        measurements = []
        for frame in frames:
            if frame < self.robot.tot_time:
                meas, _ = self.robot.get_meas(frame)
                measurements += meas
        x, y, theta = self.robot.get_gt(frames[0])
        for idx, r, b in measurements:
            if self.only_robot_measurements and idx > 5:
                continue
            dest_x = x + r*np.cos(theta + b)
            dest_y = y + r*np.sin(theta + b)
            self.measurement_lines += \
                self.ax.plot([x, dest_x],
                             [y, dest_y],
                             color=self.color)
            self.measurement_labels += \
                [plt.annotate(int(idx), (dest_x, dest_y), color=self.color)]

    def draw(self):
        """
        Add the circle and line to the axis.
        TODO does this really need its own method

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """
        self.ax.add_patch(self.circle)
        self.ax.add_line(self.line)
        if self.plot_est_pos:
            self.ax.add_patch(self.est_pos)
            self.ax.add_line(self.est_pose)
        if self.plot_landmark_uncertainty:
            for lm in self.landmark_est:
                self.ax.add_patch(lm)

    def update(self, frame):
        """
        Update the position of the robot.

        Parameters:
        ----------
        frame : int
           integer time step to plot

        Returns:
        -------
        None
        """
        x, y, theta = self.robot.get_gt(frame)
        self.x, self.y, self.theta = x, y, theta

        self.circle.set_center((self.x, self.y))
        self.line.set_data([x, x+self.wheelbase/2*np.cos(self.theta)],
                           [y, y+self.wheelbase/2*np.sin(self.theta)])
        self.label.set_position((x+self.wheelbase/2, y+self.wheelbase/2))

        if self.plot_est_pos:
            self.update_est_pos(frame)
        if self.plot_est_landmarks:
            self.update_est_landmarks(frame)
        if self.plot_measurements:
            self.update_measurements(frame)


class SceneAnimation:
    """
    This class construct an animation consisting of scene data:
     - ground-truth landmarks
     - robot positions
     - todo: robot measurements of landmarks?

    This class uses FuncAnimation from matplotlib, and includes a method
    to save the animation to a file.
    """

    def __init__(self, robots, landmark_gt, title="",
                 speedup=20, fs=50, undersample=20,
                 run_time=None,
                 plot_est_pos=True, plot_est_landmarks=True,
                 plot_landmark_uncertainty=False,
                 plot_measurements=True,
                 only_robot_measurements=False,
                 figsize=(5, 8), debug=False,
                 keys=["gt_x", "gt_y", "gt_theta"]):
        """
        Construct the animation.

        Parameters:
        ----------
        dfs: list of robot.Robot objects
           An object for each robot to plot.
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
        self.robots = robots
        self.dfs = [r.df for r in self.robots]
        self.landmark_gt = landmark_gt
        self.keys = keys
        self.title = title
        self.fs = fs
        self.length = len(self.dfs[0])  # in frames, of the actual dataset
        self.figsize = figsize
        self.plot_est_pos = plot_est_pos
        self.plot_est_landmarks = plot_est_landmarks
        self.plot_landmark_uncertainty = plot_landmark_uncertainty
        self.plot_measurements = plot_measurements
        self.only_robot_measurements = only_robot_measurements

        self.xb, self.yb = get_lims(self.dfs, landmark_gt)
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.interval = int(1000*undersample/fs/speedup)
        pause_time = 3  # seconds
        pause_frames = int(pause_time / (1000*undersample/fs/speedup))

        if run_time is None:
            self.frames = range(0, self.length + pause_frames, undersample)
        else:
            self.frames = range(0, int(run_time*fs), undersample)

        if debug:
            # don't do an animation. step through frame by frame.
            self.start_plot()
        else:
            self.ani = FuncAnimation(fig=self.fig,
                                     frames=self.frames,
                                     func=self.update_plot,
                                     init_func=self.start_plot,
                                     interval=self.interval,
                                     blit=False, repeat=False)

    def debug_event_loop(self):
        for frame in self.frames:
            print("frame")
            if frame >= self.length:
                # this must be an ending pause frame
                frame = self.length - 1
            for robot in self.robots:
                while robot.t <= frame:
                    pass
                print("moving on...")
            self.update_plot(frame)

    def start_plot(self):
        """
        Called by the constructor. Initialze the animation
        (plot the first frame).
        """

        self.anim_robots = []
        for i in range(len(self.robots)):
            r = RobotVisual(self.ax, self.robots[i],
                            name=f"{i+1}", color=colors[self.robots[i].my_idx-1],
                            plot_est_pos=self.plot_est_pos,
                            plot_est_landmarks=self.plot_est_landmarks,
                            plot_measurements=self.plot_measurements,
                            only_robot_measurements=self.only_robot_measurements,
                            plot_landmark_uncertainty=self.plot_landmark_uncertainty)
            self.anim_robots += [r]

        for i in self.landmark_gt.index:
            x, y, name = self.landmark_gt.iloc[i][
                ["x [m]", "y [m]", "Subject #"]]
            self.lm_scatter = plt.scatter(x, y, color="black")
            plt.annotate(int(name), (x, y))

        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.legend(loc="upper left")

        self.ax.autoscale_view()
        self.ax.set_xlim(*self.xb)
        self.ax.set_ylim(*self.yb)
        self.ax.axis('equal')
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

        for i in range(len(self.robots)):
            self.anim_robots[i].update(frame)
        self.ax.set_title(f"{self.title}\n{self._create_time_ctr(frame)}")
        # not sure why this has to keep getting reset..
        self.ax.set_xlim(*self.xb)
        self.ax.set_ylim(*self.yb)
        #self.ax.axis('equal')

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
        print(f"writing to {fname}. This can take a while (even after the progress bar finishes)...")
        if ".gif" in os.path.basename(fname):
            print("using Pillow GIF writer")
            writer = PillowWriter(fps=1000/self.interval)
        elif ".mp4" in os.path.basename(fname):
            print("using FFmpeg mp4 writer")
            writer = FFMpegWriter(fps=1000/self.interval)
        else:
            raise ValueError("can only write GIF or MP4 files.")
        with ProgressBar() as t:
            self.ani.save(fname, writer=writer, progress_callback=t.update_to)

    def show(self):
        plt.show()


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

class ProgressBar(tqdm):
    def update_to(self, current, total):
        self.total = total
        self.update(current - self.n)
