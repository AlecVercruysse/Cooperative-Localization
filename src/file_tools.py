import pandas as pd
import numpy as np


def read_dat(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
        header = lines[3][2:].strip().split("    ")
        arr_dat = np.genfromtxt(lines, delimiter="\t")
    df = pd.DataFrame(arr_dat, columns=header)
    return df

def get_barcode_mapping(dataset_name):
    data = read_dat(path + "Barcodes.dat")
    return  # todo


def read_experimental(dataset_name):
    # TODO do the mapping of barcode # to subject # for all instances of barcode #
    path = f"../data/{dataset_name}/"

    barcode_mapping = get_barcode_mapping(dataset_name)  # use mapping
    
    robots = [f"Robot{i}" for i in range(1, 6)]
    data = []
    for i, robotname in enumerate(robots):
        data_dict = {}
        data_dict["measurement"] = read_dat(path + f"{robotname}_Measurement.dat")
        data_dict["odometry"] = read_dat(path + f"{robotname}_Odometry.dat")
        data += [data_dict]
    return data
    
    return


def read_groundtruth(datasetnum):
    # return landmarkgt, [{robotgt}, {robotgt}, ... ]
    return
