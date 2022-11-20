import numpy as np


class EKFSLAM:
    # keep a whole separate state estimation class
    # so that this can be easily replaced
    def __init__(self, tot_time):
        """
        initialize, keep track of state and covariance for all time steps
        """
        self.tot_time = tot_time  # in units of time steps
        self.t = 0
        self.state_labels = ["x", "y", "theta"] + \
            [f"{meas}_{num+1}" for num in range(20) for meas in ["x", "y"]]

        self.state_hist = np.zeros((tot_time,
                                    len(self.state_labels)))
        self.cov_hist = np.zeros((tot_time,
                                  len(self.state_labels),
                                  len(self.state_labels)))

        # TODO
        # I feel like a lot of these functions should be implemented
        # in the robot class, then passed as arguments to the constructor?
        # to the EKF class? not sure.
        # propagation step:
        #   self.calc_prop_Gx(old_state, t)
        #   self.calc_prop_Gu(old_state, t)
        #   self.self.get_odom(t)
        # correction step:
        #   self.calc_meas_jacobian(est_state)
        #   self.calc_kalman_gain(est_cov, Ht)
        #   self.get_meas(t)
        #   self.calc_meas_prediction(est_state)
        
    def iterate(self):
        """
        move forward a single time step
        (perform a prediction and correction step).
        """
        old_state, old_cov = self.state_hist[self.t], self.cov_hist[self.t]
        self.t += 1

        est_state, est_cov = self.predict(old_state, old_cov, self.t)
        new_state, new_cov = self.correct(est_state, est_cov, self.t)
        self.state_hist[self.t] = new_state
        self.cov_hist[self.t] = new_cov

    def predict(self, old_state, old_cov, t):
        # call a function in robot class to propagate state?
        # the motion model and the data itself belongs to the robot.
        Gx = self.calc_prop_Gx(old_state, t)  # the jacobian w.r.t. state
        Gu = self.calc_prop_Gu(old_state, t)  # the jacobian w.r.t. odom
        odometry, odometry_cov = self.get_odom(t)

        state_est = Gx @ old_state + Gu @ odometry
        cov_est = (Gx @ old_cov @ Gx.T) + (Gu @ odometry_cov @ Gu.T)
        return state_est, cov_est

    def correct(self, est_state, est_cov, t):
        # call a function in robot clas to get measurements?
        # the data belongs to the robot.
        Ht = self.calc_meas_jacobian(est_state)
        Kt = self.calc_kalman_gain(est_cov, Ht)
        meas = self.get_meas(t)
        meas_prediction = self.calc_meas_prediction(est_state)

        new_state = est_state + Kt @ (meas - meas_prediction)
        new_cov = (np.identity(len(est_state)) - Kt @ Ht) @ est_cov

        return new_state, new_cov


class Robot:

    def __init__(self, df):
        self.df = 0
        self.tstep = 0  # keep track of time index
        self.state_estimator = EKFSLAM(
            tot_time=len(self.df)
        )

    def next(self, ):
        self.tstep += 1
        self.state_estimator.iterate()
