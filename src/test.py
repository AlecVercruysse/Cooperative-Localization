import visualize
import matplotlib.pyplot as plt
import pickle
import file_tools

with open("completed_slam.pkl", "rb") as f:
    robots = pickle.load(f)
dfs, landmark_gt = file_tools.get_dataset(1)

s = visualize.SceneAnimation(robots, landmark_gt, title="dataset 1", plot_est_pos=True, plot_est_landmarks=True, plot_measurements=True)
s.write("localization.gif")
# plt.show()
