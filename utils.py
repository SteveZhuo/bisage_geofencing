import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pickle
import shutil

from numpy.core.defchararray import equal
from numba.cuda import test
from scipy.sparse import data
from sklearn.metrics import classification_report, confusion_matrix

def create_folder(folder_path):
    if not (osp.exists(folder_path)):
        os.mkdir(folder_path)

def series_to_file(obj, filename):
    with open(filename, 'wb') as f_out:
        pickle.dump(obj, f_out, -1)
        print("Data written into {}".format(filename))

def file_to_series(filename):
    with open(filename, 'rb') as f_in:
        series = pickle.load(f_in)
        print("File {} loaded.".format(filename))
        return series

def rssi2weight(offset, rssi):
    return offset + rssi

def rssi2weightexp(rssi):
    return 10.0**(rssi/10.0)

def rssi2weighttrial(rssi):
    return rssi+150

def is_virtual_mac(mac_addr):
    mac_addr = mac_addr.replace(":", "").upper()
    first_hex = int(mac_addr[0:2], 16)
    return first_hex & 0x02 != 0

def interpolate_point(timestamp, timestamps, breakpoints):
    if timestamp <= timestamps[0]:
        print("timestamp too small: {} <= {}".format(timestamp, timestamps[0]))
        return breakpoints[0]
    if timestamp >= timestamps[-1]:
        print("timestamp too large: {} >= {}".format(
            timestamp, timestamps[-1]))
        return breakpoints[-1]

    for idx in range(len(timestamps)-1):
        if timestamps[idx] <= timestamp <= timestamps[idx+1]:
            return [breakpoints[idx][coor_id] + (timestamp - timestamps[idx]) /
                    (timestamps[idx+1] - timestamps[idx]) *
                    (breakpoints[idx+1][coor_id] - breakpoints[idx][coor_id])
                    for coor_id in [0, 1]]    
