import pandas as pd
import numpy as np
from tqdm import tqdm

from glob import glob
import pickle
import os
import sys
import pdb
from contextlib import contextmanager


@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    if len(path) != 0: # i.e. we're not already in that dir
        os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

def read_dat(fname):
    """
    read a single dataset .dat file

    Parameters:
    ----------
    fname: str
       name of the file to open

    Returns:
    -------
    df: pandas.DataFrame
       the data in the dat file.
    """
    with open(fname, "r") as f:
        lines = f.readlines()
        header = lines[3][2:].strip().split("    ")
        arr_dat = np.genfromtxt(lines, delimiter="\t")
    df = pd.DataFrame(arr_dat, columns=header)
    return df


def get_barcode_mapping(dataset_name):
    """
    Make a dictionary with the barcode values as keys
    and the corresponding subject number as the value.
    subjects 1-5 are the robots, 16-20 are the landmarks.

    Parameters:
    ----------
    dataset_name: str
       name of the dataset (e.g. MRCLAM_Dataset1)

    Returns:
    -------
    mapping: dict
       mapping with integer keys, integer values.
    """
    mapping = {}
    path = f"../data/{dataset_name}/"
    data = read_dat(path + "Barcodes.dat")
    for i in data.index:
        key = int(data.loc[i]["Barcode #"])
        val = int(data.loc[i]["Subject #"])
        mapping[key] = val
    return mapping


def read_experimental(dataset_name="MRCLAM_Dataset1"):
    """
    Read all measured (e.g. not ground-truth) data from a dataset.

    Parameters:
    ----------
    dataset_name: str
       name of the dataset (e.g. MRCLAM_Dataset1)

    Returns:
    -------
    data: list, length 5, of dictionaries
       a list of 5 dictionaries, each corresponding to a robot,
       with "measurement" and "odometry" as keys, corresponding
       to the measurement and odometry dat files in the dataset.
    """
    path = f"../data/{dataset_name}/"
    barcode_mapping = get_barcode_mapping(dataset_name)  # use mapping

    robots = [f"Robot{i}" for i in range(1, 6)]
    data = []
    for i, robotname in enumerate(robots):
        data_dict = {}
        data_dict["measurement"] = read_dat(path + f"{robotname}_Measurement.dat")
        data_dict["measurement"]["Barcode #"] = data_dict["measurement"]["Subject #"]
        data_dict["measurement"]["Subject #"] = data_dict["measurement"]["Subject #"].apply(
            lambda x: barcode_mapping[int(x)] if int(x) in barcode_mapping else np.nan
        )
        data_dict["odometry"] = read_dat(path + f"{robotname}_Odometry.dat")
        data += [data_dict]
    return data


def read_groundtruth(dataset_name):
    """
    Read all groundtruth data from a dataset.

    Parameters:
    ----------
    dataset_name: str
       name of the dataset (e.g. MRCLAM_Dataset1)

    Returns:
    -------
    landmark_gt: pandas.DataFrame
       A dataframe containing the ground-truth landmark locations
    data: list, length 5, of dictionaries, with single key "gt"
       The same format as the return value of `read_experimental`,
       but each dictionary contains only robot ground-truth data.
    """
    # return landmarkgt, [{robotgt}, {robotgt}, ... ]
    path = f"../data/{dataset_name}/"
    landmark_gt = read_dat(path + "Landmark_Groundtruth.dat")
    robots = [f"Robot{i}" for i in range(1, 6)]
    data = []
    for i, robotname in enumerate(robots):
        data_dict = {}
        data_dict["gt"] = read_dat(path + f"{robotname}_Groundtruth.dat")
        data += [data_dict]
    return landmark_gt, data

def get_dataset(idx, fs=50):
    """
    Return a complete dataset, with all measurement, odometry, and ground-truth
    sampled at a sampling interval of `fs`. Performs linear interpolation of
    measured data. If a particular measurement is not available at that time,
    the np.nan is used as the N/A value.

    Parameters:
    ----------
    idx: int
       A value from 1-9 describing which dataset to use.
    fs: int
       The sampling frequency in Hz.

    Returns:
    -------
    data: list, length 5, of pandas.DataFrame
       All sampled data from that dataset for each robot.
    landmark_gt: pandas.DataFrame
       ground-truth landmark location information.
    """
    try:
        with cwd(os.path.dirname(sys.argv[0])): # ensure in src/ dir
            datasets = sorted(glob("../data/*/"), key=lambda x: x[-2])
            dataset_name = datasets[idx-1]
            cache_name = dataset_name.replace("/data/", "/data/processed/") + f"{fs}.pkl"
            if os.path.exists(cache_name):
                print(f"found cache at {cache_name}")
                with open(cache_name, "rb") as f:
                    dfs = pickle.load(f)
                    landmark_gt, _ = read_groundtruth(dataset_name)
                    return dfs, landmark_gt
            else:
                print(f"no cache for this dataset & fs found, generating...")
                experimental_data = read_experimental(dataset_name)
                landmark_gt, robot_gt = read_groundtruth(dataset_name)
    except:
        raise FileNotFoundError("unable to read ground truth data from data/")
    # resample all the data to 50 Hz
    stop_time = np.inf
    start_time = -1
    for i, data_dict in enumerate(experimental_data):
        last_meas = data_dict["measurement"].loc[len(data_dict["measurement"])-1]["Time [s]"]
        start_meas = data_dict["measurement"].loc[0]["Time [s]"]

        last_odom = data_dict["odometry"].loc[len(data_dict["odometry"])-1]["Time [s]"]
        start_odom = data_dict["odometry"].loc[0]["Time [s]"]

        robot_stop_time = min(last_meas, last_odom)
        robot_start_time = max(start_meas, start_odom)

        stop_time = min(stop_time, robot_stop_time)
        start_time = max(start_time, robot_start_time)

    rel_time = np.arange(int((stop_time - start_time)*fs)) / fs
    
    # start by resampling measurements:
    # shape (5 robots, 20 measurable things, T timesteps, 2 (range, bearing))
    measurements = np.full((5, 20, len(rel_time), 2), np.nan)
    for i, robot in enumerate(experimental_data):
        print(f"processing measurements for robot {i+1}:")
        meas = robot["measurement"]

        # round measurement time to nearest sample index
        meas["idx"] = meas["Time [s]"].apply(lambda x: np.round((x - start_time)*fs))
        for idx in tqdm(meas.index):
            row = meas.iloc[idx]
            # if the measurement occurs after the starting time and before the ending time
            if row["idx"] >= 0 and row["idx"] < len(rel_time):
                if not np.isnan(row["Subject #"]): # there are some invalid barcodes...
                    measurements[i, int(row["Subject #"])-1, int(row["idx"])] = \
                        (row["range [m]"], row["bearing [rad]"])
                    
    # resample odometry and ground truth:
    
    # shape (5 robots, T timestamps, 2 (forward v, angular w))
    odometry = np.zeros((5, len(rel_time), 2))
    for i, measured in enumerate(experimental_data):
        print(f"processing odometry for robot {i+1}:")
        odom = measured["odometry"]
        for it, t in tqdm(enumerate(rel_time + start_time), total=len(rel_time)):
            idx_before = np.where(odom["Time [s]"] - t < 0)[0][-1]
            idx_after  = idx_before + 1
            # pdb.set_trace()
            mix = (t - odom["Time [s]"][idx_before]) / (odom["Time [s]"][idx_after] - odom["Time [s]"][idx_before])
            odometry[i, it] = (
                odom["forward velocity [m/s]"][idx_before] + \
                mix * (odom["forward velocity [m/s]"][idx_after] - odom["forward velocity [m/s]"][idx_before]),
                odom["angular velocity[rad/s]"][idx_before] + \
                mix * (odom["angular velocity[rad/s]"][idx_after] - odom["angular velocity[rad/s]"][idx_before])
            )
    
    # shape (5 robots, T timestamps, 3 (x, y, orientation))
    gt = np.zeros((5, len(rel_time), 3))
    for i, gtdict in enumerate(robot_gt):
        print(f"processing ground truth for robot {i+1}:")
        gt_df = gtdict["gt"]
        for it, t in tqdm(enumerate(rel_time + start_time), total=len(rel_time)):
            idx_before = np.where(gt_df["Time [s]"] - t < 0)[0][-1]
            idx_after  = idx_before + 1
            mix = (t - gt_df["Time [s]"][idx_before]) / (gt_df["Time [s]"][idx_after] - gt_df["Time [s]"][idx_before])
            gt[i, it] = (
                gt_df["x [m]"][idx_before] + \
                mix * (gt_df["x [m]"][idx_after] - gt_df["x [m]"][idx_before]),
                gt_df["y [m]"][idx_before] + \
                mix * (gt_df["y [m]"][idx_after] - gt_df["y [m]"][idx_before]),
                gt_df["orientation [rad]"][idx_before] + \
                mix * (gt_df["orientation [rad]"][idx_after] - gt_df["orientation [rad]"][idx_before])
            )
    
    # assemble dataframes:
    dfs = []
    nrobots = gt.shape[0]
    for i in range(nrobots): # in robots
        df = pd.DataFrame(np.concatenate((
            odometry[i].T, # 2, t
            np.swapaxes(measurements[i], 1, 2).reshape(2*20, len(rel_time)), # 40, (range, bearing)*20 objs
            gt[i].T
        )).T, columns = ["v", "w"] +
                          [f"{meas}_{num+1}" for num in range(20) for meas in ["r", "b"]] +
                          ["gt_x", "gt_y", "gt_theta"])
        dfs += [df]

    print("caching processed data...")
    with cwd(os.path.dirname(sys.argv[0])): # ensure in src/ dir
        if not os.path.exists(os.path.dirname(cache_name)):
            os.makedirs(os.path.dirname(cache_name))
        with open(cache_name, "wb") as f:
            pickle.dump(dfs, f)
    return dfs, landmark_gt
