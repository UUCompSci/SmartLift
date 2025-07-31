import numpy as np
import csv
import sys
import os
import math
import torch
from typing import List, Tuple
import torch
import numpy as np
from scipy.signal import find_peaks

landmark_indeces_to_labels = {
    0: "nose",
    1: "left eye inner",
    2: "left eye",
    3: "left eye outer",
    4: "right eye inner",
    5: "right eye",
    6: "right eye outer",
    7: "left ear",
    8: "right ear",
    9: "mouth left",
    10: "mouth right",
    11: "left shoulder",
    12: "right shoulder",
    13: "left elbow",
    14: "right elbow",
    15: "left wrist",
    16: "right wrist",
    17: "left pinky",
    18: "right pinky",
    19: "left index",
    20: "right index",
    21: "left thumb",
    22: "right thumb",
    23: "left hip",
    24: "right hip",
    25: "left knee",
    26: "right knee",
    27: "left ankle",
    28: "right ankle",
    29: "left heel",
    30: "right heel",
    31: "left foot index",
    32: "right foot index"
}


def calculate_angle(a,b,c):  #function only works in two dimensions as written
    a = np.array(a)  #first landmark
    b = np.array(b)  #second landmark
    c = np.array(c)  #third landmark

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


from pathlib import Path

def save_lift_data(lift_name, new_points, new_angles, filename_tag):
    # Get path to this file (e.g., functions.py), then go up one level to project root
    script_dir = Path(__file__).resolve().parent.parent
    data_dir = script_dir / 'lift data' / f"{lift_name} files"
    data_dir.mkdir(parents=True, exist_ok=True)

    points_path = data_dir / f"{lift_name} points.npz"
    angles_path = data_dir / f"{lift_name} angles.npz"

    # Load existing .npz contents if they exist
    if points_path.exists():
        existing_points = dict(np.load(points_path, allow_pickle=True))
    else:
        existing_points = {}

    if angles_path.exists():
        existing_angles = dict(np.load(angles_path, allow_pickle=True))
    else:
        existing_angles = {}

    # Add new data to existing
    for k, v in new_points.items():
        existing_points[f"{filename_tag}_{k}"] = v

    for k, v in new_angles.items():
        existing_angles[f"{filename_tag}_{k}"] = v

    # Save combined data
    np.savez(points_path, **existing_points)
    np.savez(angles_path, **existing_angles)



# Function of how to determine the side of the lift.
# Useful if we only have data for one side of the body (like in bench)
# Take the average landmark distance FOR EACH landmark, and then average
# them together
#
# essentially just taking the average of the averages
def determineSide(lift_dict, lift_name):
    #for lift_name in lift_dict:
        #print(lift_name, lift_dict[lift_name].keys())
    left_coords = []
    right_coords = []

    for i in range(11,33):

        #print(f"{' '.join(lift_name.split(' ')[:3])} landmark {i}", functions.landmark_indeces_to_labels[i])
        landmark = f"{' '.join(lift_name.split(' ')[:3])} landmark {i}"
        #print(lift_dict[lift_name][landmark])
        if "left" in landmark_indeces_to_labels[i]:
            left_coords.append(torch.mean(lift_dict[lift_name][landmark]))
        if "right" in landmark_indeces_to_labels[i]:
            right_coords.append(torch.mean(lift_dict[lift_name][landmark]))

    # uncomment to print out the left and right coord averages that the decision is based off of
    # print(np.mean(left_coords), np.mean(right_coords))

    if np.mean(left_coords) < np.mean(right_coords):
        return 'left'
    else:
        return 'right'



def get_angle_keys(lift_dict, lift_name):
    return [key for key in lift_dict[lift_name].keys() if 'landmark' not in key]


def get_point_keys(lift_dict, lift_name):
    return [key for key in lift_dict[lift_name].keys() if 'landmark' in key]


def joint_angles_to_list(lift_dict):
    angles_for_entire_lift_dict = []
    for lift_name in lift_dict:
        # Stores information for this particular lift
        # Ex. stores the name, label, and angles for 'deadlift 1 good lift data' in our DEADLIFT_TENSORS dict
        individual_lift_info = {}
        individual_lift_info['name'] = lift_name.split(' ')[0]
        individual_lift_info['viewing from'] = determineSide(lift_dict, lift_name)
        individual_lift_info['label'] = lift_name.split(' ')[2]

        # Isolate angles and store them in a separate dict to then append to individual_lift_info
        individual_lift_angles = {}
        for joint in get_angle_keys(lift_dict, lift_name):
            individual_lift_angles[" ".join(joint.split(' ')[3:5])] = lift_dict[lift_name][joint]

        individual_lift_points = {}
        for point in get_point_keys(lift_dict, lift_name):
            individual_lift_points[point] = lift_dict[lift_name][point]

        individual_lift_info['angles'] = individual_lift_angles
        individual_lift_info['points'] = individual_lift_points
        angles_for_entire_lift_dict.append(individual_lift_info)
    #print(angles_for_entire_lift_dict)
    return angles_for_entire_lift_dict



def split_lifts_into_reps(lift_list: List[dict], joint) -> List[dict]:
    rep_samples = []
    for lift in lift_list:
        viewing_from = lift['viewing from']
        motion_joint = viewing_from + " " + joint
        angles = lift['angles']
        points = lift['points']
        motion_signal = angles[motion_joint]

        rep_ranges = get_rep_ranges(motion_signal)

        for start, end in rep_ranges:
            new_sample = {
                'name': lift['name'],
                'viewing from': viewing_from,
                'label': lift['label'],
                'angles': {},
                'points': {}
            }

            for joint_name, joint_data in angles.items():
                new_sample['angles'][joint_name] = joint_data[start:end]


            for point_name, point_data in points.items():
                #print(point_name, point_data.shape)
                new_sample['points'][point_name] = ['','','']
                for row in range(len(point_data)):
                    new_sample['points'][point_name][row] = point_data[row][start:end]
            rep_samples.append(new_sample)

    return rep_samples


def get_rep_ranges(signal: torch.Tensor, prominence=10, distance=20) -> List[Tuple[int, int]]:
    y = signal.numpy()
    # Smooth the signal (optional)
    smoothed = np.convolve(y, np.ones(5)/5, mode='same')

    # Find valleys (start/end of reps)
    valleys, _ = find_peaks(-smoothed, prominence=prominence, distance=distance)

    # Backtrack a bit to capture early motion
    rep_ranges = [(max(valleys[i]-5, 0), min(valleys[i+1]+5, len(y)))
                  for i in range(len(valleys) - 1)]

    return rep_ranges