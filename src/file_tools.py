import pandas as pd
import numpy as np

from glob import glob

import pdb


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
    Mapping = {}
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
    data: pandas.DataFrame
       All sampled data from that dataset.
    """
    datasets = sorted(glob("../data/*/"), key=lambda x: x[-2])
    dataset_name = datasets[idx-1]

    experimental_data = read_experimental(dataset_name)
    landmark_gt, robot_gt = read_groundtruth(dataset_name)

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
    # rel_time + start_time = time in the data_dict

    # TODO: sample measuremnent and odometry at a specific rel_time,
    # construct new df. use linear interpolation of the two samples
    # surrounding rel_time.
