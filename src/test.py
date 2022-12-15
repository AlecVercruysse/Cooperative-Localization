import file_tools
import animate

fs = 10
dfs, landmark_gt = file_tools.get_dataset(1, fs=fs)

def meas_map_correction(idx):
    if idx == 11:
        return 17
    elif idx == 17:
        return 11
    else:
        return idx


# ground-truth ################################################################
# Just robot 2, with landmark uncertainty
animate.write_animation("../output/ground_truth.mp4", dfs, landmark_gt,
                        animation_options={"plot_est_pos": False,
                                           "plot_est_landmarks": False},
                        robot_options={"meas_map_correction": meas_map_correction},
                        title="Dataset 1")
quit()

# Scenario 1 ##################################################################
# Just robot 2, with landmark uncertainty
animate.write_animation("../output/scenario_1_robot_2.mp4", dfs, landmark_gt,
                        robot_idxs=[1],
                        animation_options={"plot_landmark_uncertainty": True},
                        robot_options={"meas_map_correction": meas_map_correction},
                        title="Scenario 1, Dataset 1, 10Hz")

# All robots, no landmark uncertainty
animate.write_animation("../output/scenario_1_all_robots.mp4", dfs, landmark_gt,
                        robot_options={"meas_map_correction": meas_map_correction},
                        title="Scenario 1, Dataset 1, 10Hz")

# All robots, no landmark uncertainty, ground-truth-initialization
animate.write_animation("../output/scenario_1_all_robots_gt_init.mp4", dfs, landmark_gt,
                        robot_options={"meas_map_correction": meas_map_correction,
                                       "gt_initialization": True},
                        title="Scenario 1, Dataset 1, 10Hz, GT init.")

# Scenario 2 ##################################################################
# All robots, no landmark uncertainty
animate.write_animation("../output/scenario_2_all_robots.mp4", dfs, landmark_gt,
                        robot_options={"meas_map_correction": meas_map_correction},
                        do_ci=True,
                        title="Scenario 2, Dataset 1, 10Hz")

# All robots, no landmark uncertainty, ground-truth-initialization
animate.write_animation("../output/scenario_1_all_robots_gt_init.mp4", dfs, landmark_gt,
                        robot_options={"meas_map_correction": meas_map_correction,
                                       "gt_initialization": True},
                        do_ci=True,
                        title="Scenario 2, Dataset 1, 10Hz, GT init.")

# Scenario 3 ##################################################################
# All robots, no landmark uncertainty
animate.write_animation("../output/scenario_3_all_robots.mp4", dfs, landmark_gt,
                        robot_options={"meas_map_correction": meas_map_correction},
                        basic_robots=[0, 2, 3, 4],
                        do_ci=True,
                        title="Scenario 3, Dataset 1, 10Hz")

# All robots, no landmark uncertainty, ground-truth-initialization
animate.write_animation("../output/scenario_3_all_robots_gt_init.mp4", dfs, landmark_gt,
                        robot_options={"meas_map_correction": meas_map_correction,
                                       "gt_initialization": True},
                        basic_robots=[0, 2, 3, 4],
                        do_ci=True,
                        title="Scenario 3, Dataset 1, 10Hz, GT init.")
