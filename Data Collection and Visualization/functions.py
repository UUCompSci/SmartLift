import numpy as np
import csv
import sys
import os
import math


def calculate_angle(a,b,c):  #function only works in two dimensions as written
    a = np.array(a)  #first landmark
    b = np.array(b)  #second landmark
    c = np.array(c)  #third landmark

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

#attempt at 3d rendering. Would need Depth Camera for more accuracy

'''def calculate_angle(a,b,c):
    a = np.array(a)  # first landmark
    b = np.array(b)  # second landmark
    c = np.array(c)  # third landmark
    # VECTORIZE POINTS
    ab = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    cb = c[0] - b[0], c[1] - b[1], c[2] - b[2]
    dot = 0
    #COMPUTE DOT PRODUCT
    for i in range(3):
        dot += ab[i] * cb[i]
    #COMPUTE NORM OF VECTORS
    abNorm = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
    cbNorm = math.sqrt(cb[0] ** 2 + cb[1] ** 2 + cb[2] ** 2)
    #CALCULATE ANGLE
    angle = np.degrees(np.arccos(dot / (abNorm * cbNorm)))
    if angle > 180.0:
        angle = 360 - angle
    return angle'''


def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(data)



from pathlib import Path

def save_lift_data(lift_name, new_points, new_angles, filename_tag):
    # Get path to this file (e.g., functions.py), then go up one level to project root
    script_dir = Path(__file__).resolve().parent.parent
    data_dir = script_dir / 'training data' / f"{lift_name} files"
    data_dir.mkdir(parents=True, exist_ok=True)

    lift_data_path = data_dir / f"{lift_name} lift data.npz"
    #angles_path = data_dir / f"{lift_name} angles.npz"

    # Load existing .npz contents if they exist
    if lift_data_path.exists():
        existing_data = dict(np.load(lift_data_path, allow_pickle=True))
    else:
        existing_data = {}

    '''if angles_path.exists():
        existing_angles = dict(np.load(angles_path, allow_pickle=True))
    else:
        existing_angles = {}'''


    # Add new data to existing
    for k, v in new_points.items():

        existing_data[f"{filename_tag}_{k}"] = v

    for k, v in new_angles.items():
        existing_data[f"{filename_tag}_{k}"] = v

    # Save combined data
    np.savez(lift_data_path, **existing_data)
    #np.savez(angles_path, **existing_angles)


