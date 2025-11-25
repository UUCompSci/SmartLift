import numpy as np
import matplotlib.pyplot as plt
import os
import mediapipe as mp
mp_pose = mp.solutions.pose

JOINTS_TO_TRACK = {
    "left_elbow": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
    "right_elbow": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
    "left_knee": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE],
    "right_knee": [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE],
    "right_hip": [mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER],
    "left_hip": [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER],
    "left_shoulder": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW],
    "right_shoulder": [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW],
}

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


#CHANGE FILES TO VIEW DIFFERENT LIFTS
angle_history = {joint: [] for joint in JOINTS_TO_TRACK}

script_dir = os.path.dirname(os.path.abspath("main.ipynb"))
videos_dir = os.path.join(script_dir, '..', 'videos')

# Normalize path just in case
videos_dir = os.path.normpath(videos_dir)
data_dir = os.path.join(script_dir, '..', 'lift data')
data = np.load(f"{data_dir}\\bench files\\bench 1 good lift data.npz")


#print(data)
# for key in data:
#     joint_data = data[key]
#     print(f"{key}: shape = {joint_data.shape}")
#     plt.plot(joint_data, label=key)
#     plt.title(f"{key} over time")
#     plt.xlabel("Frame")
#     plt.xlim(0,joint_data.shape[0])
#     plt.ylim(0,180)
#     plt.ylabel("Angle")
#
#     plt.legend()
#     plt.show()

i = 0

for key in data:
    if "landmark" in key:
        for coordinate in data[key]:
            # ISOLATE THE INDEX WHERE LANDMARK NAME IS FOUND SO WE CNA ISOLATE LANDMARK NAME
            # indexing is getting a little convoluted. Just know that this isolates the
            # landmark index and maps it to the landmark name
            print(key)
            s = key.find("landmark_")
            m = key.find(".")

            current_landmark_index = int(key[s+9:])
            current_landmark_name = landmark_indeces_to_labels[current_landmark_index]

            plt.plot(coordinate, label=chr(120+i))
            i += 1
            if i == 3:
                i = 0

        plt.title(f"{key[:m]} {current_landmark_name} coordinates over time")
        plt.xlabel("Frame")
        plt.ylabel('Coordinate Position on Screen')
        plt.ylim(-1,1)
        file_path = os.path.join("visualizations", f"{key}.png")
        plt.savefig(file_path)
        plt.legend()

        # Toggle on or off depending on if you want to see them on screen or just save them
        plt.show()

    else:
        joint_data = data[key]
        print(f"{key}: shape = {joint_data.shape}")
        plt.plot(joint_data, label=key)
        plt.title(f"{key} over time")
        plt.xlabel("Frame")
        plt.xlim(0,joint_data.shape[0])
        plt.ylim(0,180)
        plt.ylabel("Angle")

        plt.legend()
        file_path = os.path.join("visualizations", f"{key}.png")
        plt.savefig(file_path)

        plt.show()