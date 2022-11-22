import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np

import pdb
import os
import code
import file_tools
import ekf_tools
colors = ["red", "orange", "lime", "cyan", "orchid"]

dfs, landmark_gt = file_tools.get_dataset(1)
robots = [ekf_tools.Robot(df, fs=50, landmark_gt=landmark_gt)
              for df in dfs]

class Error:

##TODO: take list of robots from SceneAnimation, take get_gt for groundtruth, get estimate position, 
# calculate mean square error for each time step for each robot

    def error_calc(robots):
        ekf_tools.get_gt(self,t)
        ekf_tools.get_est_pos(self,t)
        return
