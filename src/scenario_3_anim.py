import file_tools
from tqdm import tqdm
import matplotlib.pyplot as plt

import pdb
import code

from visualize import SceneAnimation
import robot


def meas_map_correction(idx):
    if idx == 11:
        return 17
    elif idx == 17:
        return 11
    else:
        return idx


fs=10
dfs, landmark_gt = file_tools.get_dataset(1, fs=fs)    
robots = [robot.Robot(df, fs=fs, landmark_gt=landmark_gt, my_idx=i+1,
                      gt_initialization=False,
                      meas_map_correction=meas_map_correction,
                      basic_robot=(i > 0)
                      )
          for i, df in enumerate(dfs)]

robots = [robots[1], robots[4]]  # keep 2 and 5
# robots = [robots[4]]  # keep 2 and 5

for robot in robots:
    other_robots = robots.copy()
    other_robots.remove(robot)
    robot.other_robots = other_robots

for t in tqdm(range(800)):
# for t in tqdm(range(robots[0].tot_time - 1)):
    for r in robots:
        r.next()

s = SceneAnimation(robots, landmark_gt, title="EKF-SLAM w/comms, gt init.",
                   plot_est_pos=True, plot_est_landmarks=True,
                   plot_landmark_uncertainty=True,
                   plot_measurements=True, run_time=80,
                   undersample=5, speedup=5, fs=fs)

print("\n\ncreated s, an animation object. try either" +
      "\ns.write(\"out.gif\") or \nplt.show()\n\n")
code.interact(local=locals())
        # s.write("out.gif")
        # plt.show()
