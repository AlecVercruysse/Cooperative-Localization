import numpy as np
from tqdm import tqdm

import pdb
import code

import robot

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

from rich import print, pretty
pretty.install()


# weird issue with non responsive plots when using the default
# mac backend... this is not an issue on linux.
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")
    
fs = 10
dfs, landmark_gt = file_tools.get_dataset(1, fs=fs)
# r = Robot(dfs[0], fs=fs, landmark_gt=landmark_gt)
robots = [robot.Robot(df, fs=fs, landmark_gt=landmark_gt, my_idx=i+1) for i, df in enumerate(dfs)]
for robot in robots:
    other_robots = robots.copy()
    other_robots.remove(robot)
    robot.other_robots = other_robots

scene = visualize.SceneAnimation(robots, landmark_gt, title="EKF SLAM",
                                 plot_est_pos=True,
                                 plot_est_landmarks=True,
                                 plot_measurements=True,
                                 debug=True, fs=fs)
plt.ion()
plt.show()
print("At each time step, press <ENTER> to move to the next," +
      " or <i> then <ENTER> to start an interactive terminal")
total_time = robots[0].tot_time - 1
for t in tqdm(range(total_time)):
    for r in robots:
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
                for r in robots:
                    r.next(debug=True, callback=scene.update_plot)
        except:
            pass
