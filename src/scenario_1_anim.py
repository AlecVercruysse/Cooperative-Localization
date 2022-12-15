import file_tools
from tqdm import tqdm

from visualize import SceneAnimation
import robot

fs = 10
dfs, landmark_gt = file_tools.get_dataset(1, fs=fs)

# All Robots #########################################################
robots = [robot.Robot(df,
                      fs=fs, landmark_gt=landmark_gt,
                      gt_initialization=False,
                      my_idx=i+1)
          for i, df in enumerate(dfs)]

for t in tqdm(range(robots[0].tot_time - 1)):
    for r in robots:
        r.next()

s = SceneAnimation(robots, landmark_gt, title="Scenario 1, Dataset 1, 10Hz",
                   plot_est_pos=True, plot_est_landmarks=True,
                   # plot_landmark_uncertainty=True,
                   plot_measurements=True, run_time=None,
                   undersample=20, speedup=20, fs=fs)

s.write("../output/scenario_1_all_robots.mp4")

# All Robots, Gt initialization #####################################
robots = [robot.Robot(df,
                      fs=fs, landmark_gt=landmark_gt,
                      gt_initialization=True,
                      my_idx=i+1)
          for i, df in enumerate(dfs)]

for t in tqdm(range(robots[0].tot_time - 1)):
    for r in robots:
        r.next()

s = SceneAnimation(robots, landmark_gt, title="Scenario 1, Dataset 1, 10Hz",
                   plot_est_pos=True, plot_est_landmarks=True,
                   # plot_landmark_uncertainty=True,
                   plot_measurements=True, run_time=None,
                   undersample=20, speedup=20, fs=fs)

s.write("../output/scenario_1_all_robots.mp4")


# Robot 2 ###########################################################
robots = [robot.Robot(dfs[1], fs=fs,
                      landmark_gt=landmark_gt,
                      gt_initialization=False, my_idx=2)]

for t in tqdm(range(robots[0].tot_time - 1)):
    for r in robots:
        r.next()

s = SceneAnimation(robots, landmark_gt, title="Scenario 1, Dataset 1, 10Hz",
                   plot_est_pos=True, plot_est_landmarks=True,
                   plot_landmark_uncertainty=True,
                   plot_measurements=True, run_time=None,
                   undersample=20, speedup=20, fs=fs)
s.write("../output/scenario_1_robot_2.mp4")
