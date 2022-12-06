import numpy as np
from tqdm import tqdm

import pdb
import code

import robot

import file_tools
import visualize
import matplotlib.pyplot as plt
import sys

# weird issue with non responsive plots when using the default
# mac backend... this is not an issue on linux.
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")

def angle_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

fs = 10
dfs, landmark_gt = file_tools.get_dataset(1, fs=fs)
#r = Robot(dfs[0], fs=fs, landmark_gt=landmark_gt)
robots = [robot.Robot(df, fs=fs, landmark_gt=landmark_gt) for df in dfs]

all_robots_errors = []

for robot_ind, robot in enumerate(robots):
    if(robot_ind) >= 1: ###################################### REMOVE ####################### TO PROCESS ALL ROBOTS ##########
        break
    robot_errors = []
    print(f'Processing errors from robot {robot_ind}')
    for time in tqdm(range(robot.tot_time), total = robot.tot_time):
        all_r_meas, all_r_meas_cov = robot.get_meas(time)
        if len(all_r_meas) == 0:
            continue
        else:
            for r_meas in all_r_meas:
                subject_meas = r_meas[0]
                if subject_meas <= 5 or subject_meas == 11 or subject_meas == 17:
                    continue

                r_gt = robot.get_gt(time)
                l_gt = landmark_gt[landmark_gt['Subject #'] == subject_meas] # TODO: swap first 0 index with a loop
                pos_diff = l_gt.values[0, 1:3] - r_gt.values[0:2]

                l_world_bearing_gt = np.arctan2(pos_diff[1], pos_diff[0])
                l_range_gt = (pos_diff[1]**2 + pos_diff[0]**2)**0.5

                l_bearing_meas = r_meas[2]
                l_world_bearing_meas = angle_wrap(r_gt['gt_theta'] + l_bearing_meas)
                l_range_meas = r_meas[1]

                bearing_error = angle_wrap(l_world_bearing_gt - l_world_bearing_meas)
                range_error = l_range_gt - l_range_meas

                all_robots_errors.append([range_error, bearing_error])

    #robot_errors = np.array(robot_errors)
    #all_robots_errors.append(robot_errors)

all_robots_errors = np.array(all_robots_errors)

RMSE = (np.sum(all_robots_errors**2, axis = 0)/all_robots_errors.shape[0])**0.5
print(f'RMS range_error: {RMSE[0]} m | RMS bearing_error: {RMSE[1]} rads')

fig, ax = plt.subplots(1,2, figsize = (20,6))
ax[0].set_title('Range Error Histogram (m)')
ax[0].hist(np.abs(all_robots_errors[:,0]), bins = 100)
ax[0].set_xlabel('Error (m)')
ax[0].set_ylabel('Counts')

ax[1].set_title('Bearing Error Histogram (rads)')
ax[1].hist(np.abs(all_robots_errors[:,1]), bins = 100)
ax[1].set_xlabel('Error (rads)')
ax[1].set_ylabel('Counts')
plt.show()
