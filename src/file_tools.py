import pandas as pd
import numpy as np

from glob import glob

import pdb


def read_dat(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
        header = lines[3][2:].strip().split("    ")
        arr_dat = np.genfromtxt(lines, delimiter="\t")
    df = pd.DataFrame(arr_dat, columns=header)
    return df


def get_barcode_mapping(dataset_name):
    mapping = {}
    path = f"../data/{dataset_name}/"
    data = read_dat(path + "Barcodes.dat")
    for i in data.index:
        key = int(data.loc[i]["Barcode #"])
        val = int(data.loc[i]["Subject #"])
        mapping[key] = val
    return mapping


def read_experimental(dataset_name="MRCLAM_Dataset1"):
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
