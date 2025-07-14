import numpy as np
import csv
import sys
import os
import math
import torch

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
    data_dir = script_dir / 'testing data' / f"{lift_name} files"
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

def determineSide(z_coordinates: dict, lift_to_examine: str):
    left_coords = []
    right_coords = []


    for i in range(11, 32):

        if "left" in landmark_indeces_to_labels[i]:
            #print(f'{lift_to_examime} {landmark_indeces_to_labels[i]} z coordinates', torch.mean(z_coordinates[f'{lift_to_examime} {landmark_indeces_to_labels[i]} z coordinates']))
            left_coords.append(torch.mean(z_coordinates[f'{lift_to_examine} {landmark_indeces_to_labels[i]} z coordinates']).item())
        if "right" in landmark_indeces_to_labels[i]:
            #print(f'{lift_to_examime} {landmark_indeces_to_labels[i]} z coordinates', torch.mean(z_coordinates[f'{lift_to_examime} {landmark_indeces_to_labels[i]} z coordinates']))
            right_coords.append(torch.mean(z_coordinates[f'{lift_to_examine} {landmark_indeces_to_labels[i]} z coordinates']).item())
    print("The lower the z coordinate, the closer to the camera")
    print("Mean of the mean left side landmark z distance: " + str(np.mean(left_coords)))
    print("Mean of the mean right side landmark z distance: " + str(np.mean(right_coords)))
    if np.mean(left_coords) < np.mean(right_coords):
        return 'viewing from left'
    else:
        return 'viewing from right'


def get_angle_keys(lift_dict, lift_name):
    return [key for key in lift_dict[lift_name].keys() if 'landmark' not in key]


def joint_angles_to_list(lift_dict):
    angles_for_entire_lift_dict = []
    for lift_name in lift_dict:
        # Stores information for this particular lift
        # Ex. stores the name, label, and angles for 'deadlift 1 good lift data' in our DEADLIFT_TENSORS dict
        individual_lift_info = {}
        individual_lift_info['name'] = lift_name.split(' ')[0]

        print(" ".join(lift_name.split(' ')[0:3]))

        individual_lift_info['label'] = lift_name.split(' ')[2]
        individual_lift_angles = {}
        for joint in get_angle_keys(lift_dict, lift_name):
            individual_lift_angles[" ".join(joint.split(' ')[3:5])] = lift_dict[lift_name][joint]
            # print(lift_name, joint, lift_dict[lift_name][joint])
            # print(" ".join(joint.split(' ')[3:5]))
        individual_lift_info['angles'] = individual_lift_angles
        angles_for_entire_lift_dict.append(individual_lift_info)
    #print(angles_for_entire_lift_dict)
    return angles_for_entire_lift_dict