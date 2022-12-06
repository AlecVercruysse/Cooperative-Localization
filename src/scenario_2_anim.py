import file_tools
from tqdm import tqdm
import matplotlib.pyplot as plt

import pdb
import code

from visualize import SceneAnimation
import robot

fs=10
dfs, landmark_gt = file_tools.get_dataset(1, fs=fs)
robots = [robot.Robot(df, fs=fs, landmark_gt=landmark_gt, my_idx=i+1) for i, df in enumerate(dfs)]

for robot in robots:
    other_robots = robots.copy()
    other_robots.remove(robot)
    robot.other_robots = other_robots

for t in tqdm(range(robots[0].tot_time - 1)):
    for r in robots:
        r.next()

s = SceneAnimation(robots, landmark_gt, title="Dataset 1, EKF-SLAM (guessing cov)",
                   plot_est_pos=True, plot_est_landmarks=True,
                   plot_measurements=True,
                   undersample=20, speedup=20, fs=fs)

print("\n\ncreated s, an animation object. try either" +
      "\ns.write(\"out.gif\") or \nplt.show()\n\n")
code.interact(local=locals())
        # s.write("out.gif")
        # plt.show()
