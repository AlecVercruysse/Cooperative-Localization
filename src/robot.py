import numpy as np
from tqdm import tqdm

import pdb
import code

from ekf import EKFSLAM

class Robot:

    def __init__(self, df, fs, landmark_gt=None):
        self.df = df
        self.tot_time = len(df)
        self.dt = 1/fs
        self.t = 0  # keep track of time index
        self.landmark_gt = landmark_gt
        self.state_estimator = EKFSLAM(
            robot=self
        )

    def get_odom(self, t):
        """
        get odometry info at a timestep.

        Parameters:
        ----------
        t : int
           time index

        Returns:
        -------
        odom : tuple of 2 floats
           v (forward speed in m/s) and omega (angular speed in rad/s)
        odom_cov : 2x2 np.array
           covariance matrix of odometry estimates.
        """
        odom = self.df.iloc[t][["v", "w"]]
        odom_cov = np.eye(len(odom)) * 0.4  # TODO odometry covariance!
        return odom, odom_cov

    def get_meas(self, t):
        """
        Get all measurements at a timestep.
        Returns measurements of other robots as well.

        Parameters:
        ----------
        t : int
           time index

        Returns:
        -------
        meas : list of 3-tuples
           list of tuples containing (subject #, range, bearing)
           for all measurements at that time stamp. Might be of length zero.
        meas_cov : 2x2 array
           covariance matrix of (range, bearing) uncertainty.
        """
        s = self.df.iloc[t]
        meas = [(i+1, s[f"r_{i+1}"], s[f"b_{i+1}"])
                for i in range(20)
                if not np.isnan(s[f"r_{i+1}"])]
        meas_cov = np.eye(2) * 0.2  # TODO!!!!!!!
        return meas, meas_cov

    def get_gt(self, t):
        """
        Get ground-truth position information.
        """
        return self.df.iloc[t][["gt_x", "gt_y", "gt_theta"]]

    def get_landmark_gt(self, idx):
        if self.landmark_gt is None:
            return None
        else:
            df_idx = self.landmark_gt.index[
                self.landmark_gt["Subject #"] == idx]
            return self.landmark_gt.iloc[df_idx][["x [m]", "y [m]"]].iloc[0]

    def get_est_pos(self, t):
        """
        Get estimated position information.
        """
        return self.state_estimator.get_est_pos(t)

    def get_est_landmark(self, t, idx):
        """
        Get estimated landmark information at time t.
        """
        return self.state_estimator.get_est_landmark(t, idx)

    def next(self, callback=lambda x: x, debug=False):
        """
        perform an interation.

        `callback` is an optional function that is called on
        completion of the iteration. It must take the current
        time step (after the iteration) as an argument.

        The `debug` flag is passed to the state estimator
        and controls some debug printing on stdout.
        """
        self.t += 1
        n_corrections = self.state_estimator.iterate(debug=debug)
        callback(self.t)
        return n_corrections


if __name__ == "__main__":
    # run an interactive version of this state estimation.
    # press <enter> in the repl to advance to the next time step.
    #
    # type a number and press enter in the repl to advance that
    # amount of time steps.
    #
    # type the `i` character then enter in the repl to enter
    # interactive mode (to e.g. call pdb.set_trace(), exit
    # interactive mode, then step through a function).
    import file_tools
    import visualize
    import matplotlib.pyplot as plt
    import sys

    # weird issue with non responsive plots when using the default
    # mac backend... this is not an issue on linux.
    if sys.platform == "darwin":
        import matplotlib
        matplotlib.use("TkAgg")

    fs = 10
    dfs, landmark_gt = file_tools.get_dataset(1, fs=fs)
    r = Robot(dfs[0], fs=fs, landmark_gt=landmark_gt)
    scene = visualize.SceneAnimation([r], landmark_gt, title="EKF SLAM",
                                     plot_est_pos=True,
                                     plot_est_landmarks=True,
                                     plot_measurements=True,
                                     debug=True)
    plt.ion()
    plt.show()
    print("At each time step, press <ENTER> to move to the next," +
          " or <i> then <ENTER> to start an interactive terminal")
    for t in tqdm(range(r.tot_time-1)):
        n = r.next(debug=True, callback=scene.update_plot)
        text_in = input()
        if text_in == "i":
            code.interact(local=locals())
        elif len(text_in) != 0:
            try:
                n = int(text_in)
                print(f"moving forward by {n} steps...")
                for i in range(n):
                    t += 1
                    r.next(debug=True, callback=scene.update_plot)
            except:
                pass
