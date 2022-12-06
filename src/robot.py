import numpy as np
from tqdm import tqdm

import pdb
import code

from ekf import EKFSLAM

class Robot:

    def __init__(self, df, fs, landmark_gt=None, other_robots=[], my_idx=None, gt_initialization=False):
        """
        Other_robots is a list of other robot objects. This list contains
        the other robots that this robot will query to see if they have measurement
        data.

        my_idx only matters if other_robots is not empty.
        """
        self.df = df
        self.tot_time = len(df)
        self.dt = 1/fs
        self.t = 0  # keep track of time index
        self.landmark_gt = landmark_gt
        self.state_estimator = EKFSLAM(
            robot=self,
            gt=gt_initialization
        )
        self.other_robots = other_robots
        self.my_idx = my_idx

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

    def next(self, callback=lambda x: x, debug=False, correct=True):
        """
        perform an interation.

        `callback` is an optional function that is called on
        completion of the iteration. It must take the current
        time step (after the iteration) as an argument.

        The `debug` flag is passed to the state estimator
        and controls some debug printing on stdout.
        """
        self.t += 1
        n_corrections = self.state_estimator.iterate(debug=debug,correct=correct)
        callback(self.t)
        return n_corrections
