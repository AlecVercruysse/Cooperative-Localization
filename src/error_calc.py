import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

import pdb
import os
import file_tools
import robot
colors = ["red", "orange", "lime", "cyan", "orchid"]


def angle_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

fs = 1
dfs, landmark_gt = file_tools.get_dataset(1, fs = fs)

dfs = dfs[0:4]

odom_robots = [robot.Robot(df, fs=fs, landmark_gt=landmark_gt) for df in dfs]
print('Running odom only state estimation')
for t in tqdm(range(odom_robots[0].tot_time - 1)):
    for r in odom_robots:
        r.next(correct = False)

ekf_robots = [robot.Robot(df, fs=fs, landmark_gt=landmark_gt) for df in dfs]
print('Running EKF SLAM state estimation')
for t in tqdm(range(ekf_robots[0].tot_time - 1)):
    for r in ekf_robots:
        r.next(correct = True)

ci_robots = [robot.Robot(df, fs=fs, landmark_gt=landmark_gt, my_idx=i+1) for i, df in enumerate(dfs)]
for robot in ci_robots:
    other_robots = ci_robots.copy()
    other_robots.remove(robot)
    robot.other_robots = other_robots

print('Running CI state estimation')
for t in tqdm(range(ci_robots[0].tot_time - 1)):
    for r in ci_robots:
        r.next()

def compute_errors(robots):
    """
    computes x,y,bearing error for a list of robots at all timesteps.
    I.e. state_est - groundtruth

    Parameters:
    ----------
    robots: list of robot objects

    Returns:
    -------
    all_errors: np array of size (timesteps, robots, 3)
    all_errors[i,j,:] = [x_error, y_error, bearing_error] for ith robot at jth timestep
    """
    select_times = np.arange(0, robots[0].tot_time, 1)
    all_errors = np.zeros((select_times.shape[0], len(robots), 3))
    for robot_ind, robot in enumerate(robots):
        print(f'Processing robot: {robot_ind}')
        for time_ind, time in enumerate(select_times):
            pos_est, cov = robot.get_est_pos(time)
            pos_gt = robot.get_gt(time)
            all_errors[time_ind, robot_ind, :] = pos_gt - pos_est
    all_errors[:,:,2] = angle_wrap(all_errors[:,:,2])
    return select_times, all_errors

odom_robot_times, odom_robot_errors = compute_errors(odom_robots)

ekf_robot_times, ekf_robot_errors = compute_errors(ekf_robots)

ci_robot_times, ci_robot_errors = compute_errors(ci_robots)

all_robots_res_dict = {'ekf'  :
                            {'times'  : ekf_robot_times,
                             'errors' : ekf_robot_errors,
                             'robots' : ekf_robots},
                       'ci'   :
                            {'times'  : ci_robot_times,
                             'errors' : ci_robot_errors,
                             'robots' : ci_robots}
                      }

# 'odom' :
#                             {'times'  : odom_robot_times,
#                              'errors' : odom_robot_errors,
#                              'robots' : odom_robots},

fig, ax = plt.subplots(len(dfs),2, figsize = (20,10))

for robot_ind in range(len(dfs)):
    ax[robot_ind, 0].set_title('Position error over time (m)')
    for key in all_robots_res_dict.keys():
        robot_res_dict = all_robots_res_dict[key]
        robots = robot_res_dict['robots']
        robot_times = robot_res_dict['times']
        robot_errors = robot_res_dict['errors']
        ax[robot_ind, 0].plot(robot_times * robots[robot_ind].dt,
               np.linalg.norm(robot_errors[:,robot_ind,0:2], axis = 1))
        ax[robot_ind, 1].plot(robot_times * robots[robot_ind].dt,
                              robot_errors[:,robot_ind,2])
    ax[robot_ind, 0].set_xlabel('time (s)')
    ax[robot_ind, 0].set_ylabel('error (m)')

    ax[robot_ind, 1].set_title('Orientation error over time(rads)')

    ax[robot_ind, 1].set_xlabel('time (s)')
    ax[robot_ind, 1].set_ylabel('error (rads)')

plt.show()

pdb.set_trace()

#
# class Error:
#
# ##TODO: take list of robots from SceneAnimation, take get_gt for groundtruth, get estimate position,
# # calculate mean square error for each time step for each robot
#
#     def error_calc(robots):
#         ekf_tools.get_gt(self,t)
#         ekf_tools.get_est_pos(self,t)
#         return
