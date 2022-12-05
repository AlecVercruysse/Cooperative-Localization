import numpy as np
from tqdm import tqdm

import pdb
import code


def angle_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class EKFSLAM:
    # keep a whole separate state estimation class
    # so that this can be easily replaced
    def __init__(self, robot, gt=False):
        """
        initialize, keep track of state and covariance for all time steps.

        gt is a boolean that defines whether the robot is initialized with
        ground truth data. If not, the robot initializes itself at ground truth
        (so that the estimator has the global coordinate reference frame) but
        all landmarks are initialized with their measured value, the first time
        they are observed. 
        """
        self.robot = robot
        self.t = 0
        self.state_labels = ["x", "y", "theta"] + \
            [f"{meas}_{num+1}" for num in range(5, 20) for meas in ["x", "y"]]

        self.state_hist = np.zeros((self.robot.tot_time,
                                    len(self.state_labels)))
        self.cov_hist = np.zeros((self.robot.tot_time,
                                  len(self.state_labels),
                                  len(self.state_labels)))

        # use one-indexed landmark idx (a couple indeces are unused).
        self.landmark_seen = [False for _ in range(20 + 1)]

        if gt:
            print("warning: using gt for initialization")
            # pdb.set_trace()
            gt_x, gt_y, gt_theta = self.robot.get_gt(0)
            self.state_hist[0] = [gt_x, gt_y, gt_theta] + \
                [x for num in range(5, 20)
                 for x in np.array(self.robot.get_landmark_gt(num+1))]
            self.cov_hist[0] = np.eye(len(self.state_labels)) * \
                np.array([0.1 if i+1 >= 6 else 1
                          for i in range(len(self.state_labels))])
        else:
            print("using realistic landmark initialization")
            gt_x, gt_y, gt_theta = self.robot.get_gt(0)
            self.state_hist[0][:3] = (gt_x, gt_y, gt_theta)
            self.cov_hist[0] = np.eye(len(self.state_labels)) * \
                np.array([1 if i+1 >= 6 else .1
                          for i in range(len(self.state_labels))])

    def calc_prop_Gx(self, old_state, t):
        """
        Calculate the linearization of the the propagation function
        with respect to the previous state. This is used in calculating
        the covariance of the propagated state in the prediction step.
        """
        (v, _), _ = self.robot.get_odom(t)  # need v
        # old_theta = old_state[2]

        # dx, dy, dtheta in terms of dx, dy, dtheta.
        # see paper, eq (9)
        robot_Gx = np.array([
            [1, 0, 0],  # -v*np.sin(old_theta)*self.robot.dt],
            [0, 1, 0],  # v*np.sin(old_theta)*self.robot.dt],
            [0, 0, 1]
        ])

        # mapping does not change in prediction.
        Gx = np.eye(len(self.state_labels))
        Gx[0:3, 0:3] = robot_Gx
        return Gx

    def calc_prop_Gu(self, old_state, t):
        """
        Calculate the linearization of the the propagation function
        with respect to the control input. This is used in calculating
        the covariance of the propagated state in the prediction step.
        """
        old_theta = old_state[2]

        Gu = np.zeros((len(self.state_labels), 2))
        # see paper, eq (10)
        robot_Gu = np.array([
            [np.cos(old_theta)*self.robot.dt, 0],
            [np.sin(old_theta)*self.robot.dt, 0],
            [0, self.robot.dt]
        ])

        # mapping does not change in prediction.
        Gu[0:3, :] = robot_Gu
        return Gu

    def calc_meas_jacobian(self, est_state, landmark_idx):
        """
        For a measurement to a single landmark...
        the meas model converts robot and landmark location
        into a range, bearing measurement.
        Ht linearizes that model, found in eq. (11).

        est_state: a vector describing the current state estimate
        landmark_idx: an integer describing the index of the landmark,
                      as described in the dataset.
        """
        x, y, theta, = est_state[[0, 1, 2]]
        lx_idx = self.state_labels.index(f"x_{landmark_idx}")
        ly_idx = self.state_labels.index(f"y_{landmark_idx}")
        lx, ly = est_state[lx_idx], est_state[ly_idx]

        Ht = np.zeros((2, len(self.state_labels)))

        # add some eps=1e-9 to guard against divide by zero.
        r = np.sqrt((lx - x)**2 + (ly - y)**2) + 1e-9

        drdx = (x - lx) / r
        drdy = (y - ly) / r
        drdt = 0

        dbdx = (ly - y) / r**2
        dbdy = (x - lx) / r**2
        dbdt = -1

        Ht_robot = np.array([
            [drdx, drdy, drdt],
            [dbdx, dbdy, dbdt]
        ])

        drdlx = (lx - x) / r
        drdly = (ly - y) / r

        dbdlx = (y - ly) / r**2
        dbdly = (lx - x) / r**2

        Ht_landmark = np.array([
            [drdlx, drdly],
            [dbdlx, dbdly]
        ])
        Ht[:, 0:3] = Ht_robot
        Ht[:, lx_idx:ly_idx+1] = Ht_landmark
        return Ht

    def calc_meas_prediction(self, est_state, landmark_idx):
        """
        calculate the predicted (range, bearing) of a measurement
        given an estimated state and the index of the landmark
        being measured.
        """
        x, y, theta, = est_state[[0, 1, 2]]
        lx_idx = self.state_labels.index(f"x_{landmark_idx}")
        ly_idx = self.state_labels.index(f"y_{landmark_idx}")
        lx, ly = est_state[lx_idx], est_state[ly_idx]
        r = np.sqrt((lx - x)**2 + (ly - y)**2)
        b = np.arctan2(ly - y, lx - x) - theta
        return np.array([r, b])

    def iterate(self, debug=False):
        """
        move forward a single time step
        (perform a prediction and correction step).
        """
        old_state, old_cov = self.state_hist[self.t], self.cov_hist[self.t]
        self.t += 1

        est_state, est_cov = self.predict(old_state, old_cov, self.t,
                                          debug=debug)
        new_state, new_cov, n_corrections = self.correct(est_state,
                                                         est_cov, self.t,
                                                         debug=debug)
        self.state_hist[self.t] = new_state
        self.cov_hist[self.t] = new_cov
        return n_corrections

    def g(self, old_state, odometry):
        """
        The nonlinear state propagation function.
        """
        #pdb.set_trace()
        est_state = np.copy(old_state)
        x, y, theta = old_state[:3]
        est_state[:3] = old_state[:3] + \
            np.array([
                [np.cos(theta) * self.robot.dt, 0],
                [np.sin(theta) * self.robot.dt, 0],
                [0, self.robot.dt],
            ]) @ odometry
        return est_state

    def predict(self, old_state, old_cov, t, debug=False):
        """
        Prediction step.
        """
        Gx = self.calc_prop_Gx(old_state, t)  # the jacobian w.r.t. state
        Gu = self.calc_prop_Gu(old_state, t)  # the jacobian w.r.t. odom
        odometry, odometry_cov = self.robot.get_odom(t)

        state_est = self.g(old_state, odometry)
        cov_est = (Gx @ old_cov @ Gx.T) + (Gu @ odometry_cov @ Gu.T)

        state_est[2] = angle_wrap(state_est[2])

        if not np.allclose(old_state[3:], state_est[3:]):
            # debug why landmarks 6, 7 are getting moved
            pdb.set_trace()

        if debug:
            print(f"{odometry=} \n\n {Gx=} \n\n {Gu=} \n\n {state_est=}")
        return state_est, cov_est

    def correct(self, est_state, est_cov, t, debug=False):
        """
        Correction step. Note that this might perform zero corrections,
        and might perform many. It depends on how many measurements
        we get in that time step.

        The first time we see a landmark, we initialize that landmark.
        So we do not perform a correction step with it.
        """
        meas, meas_cov = self.robot.get_meas(t)
        for landmark in meas:
            # run the correction step as many times as there are measurements
            lidx, r, b = landmark
            # omitting 11 and 17 because these measurements are switched!
            if lidx <= 5 or lidx == 11 or lidx == 17:
                continue  # we're not using robot measurements


            if not self.landmark_seen[lidx]:
                # perform first-time initialization:
                lx_idx = self.state_labels.index(f"x_{lidx}")
                ly_idx = self.state_labels.index(f"y_{lidx}")
                x, y, theta = est_state[:3]
                est_state[lx_idx] = x + r * np.cos(theta + b)
                est_state[ly_idx] = y + r * np.sin(theta + b)
                self.landmark_seen[lidx] = True
                return est_state, est_cov, 0

            # pdb.set_trace()
            measurement = np.array([r, b])
            Ht = self.calc_meas_jacobian(est_state, lidx)
            meas_prediction = self.calc_meas_prediction(est_state, lidx)

            Kt = est_cov @ Ht.T @ np.linalg.inv(Ht @ est_cov @ Ht.T + meas_cov)
            est_state = est_state + Kt @ (measurement - meas_prediction)
            est_cov = (np.identity(len(est_state)) - Kt @ Ht) @ est_cov
            est_state[2] = angle_wrap(est_state[2])

        return est_state, est_cov, len(meas)

    def get_est_pos(self, t):
        """
        return a tuple of predicted ([x, y, heading], covariance)
        for the robot at time step t.
        """
        return self.state_hist[t, 0:3], self.cov_hist[t, 0:3, 0:3]

    def get_est_landmark(self, t, idx):
        """
        return a tuple of predicted ([landmark x, landmark y], covariance)
        for the landmark `idx`'s position as estimated by the robot at time
        step t.
        """
        lx_idx = self.state_labels.index(f"x_{idx}")
        ly_idx = self.state_labels.index(f"y_{idx}")
        state = self.state_hist[t][[lx_idx, ly_idx]]
        cov = self.cov_hist[t][np.ix_([lx_idx, ly_idx],
                                      [lx_idx, ly_idx])]
        return state, cov


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

    dfs, landmark_gt = file_tools.get_dataset(1)
    r = Robot(dfs[0], fs=50, landmark_gt=landmark_gt)
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
