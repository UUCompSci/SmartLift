# Make sure "show plots in tool window" is disabled in settings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

script_dir = os.path.dirname(os.path.abspath("main.ipynb"))
videos_dir = os.path.join(script_dir, '..', 'videos')
# Normalize path just in case
videos_dir = os.path.normpath(videos_dir)
data_dir = os.path.join(script_dir, '..', 'training data')
data = np.load(f"{data_dir}\\deadlift files\\deadlift lift data.npz")

for key in data:
    if "landmark" not in key:

        joint_data = data[key]
        frames = range(10, len(joint_data) + 1, 10)  # +1 to include full length

        print(f"Processing {key}: shape = {joint_data.shape}")

        fig, ax = plt.subplots()
        ax.set_xlim(0, len(joint_data))
        ax.set_ylim(0, 180)
        ax.set_xlabel("Frame", fontsize=20)
        ax.set_ylabel("Angle", fontsize=20)
        ax.set_title(f"{key} over time", fontsize=20)  # or any size you want

        line, = ax.plot([], [], label=key)
        ax.legend()
        ax.grid(True)

        lm = key.split("_")
        key = lm[1] + " " + lm[2]

        def update(frame):
            frame = min(frame, len(joint_data))
            line.set_data(range(frame), joint_data[:frame])
            ax.set_title(f"{key} over time", fontsize=20)
            return line,


        frames = range(10, len(joint_data) + 10, 5)
        anim = FuncAnimation(fig, update, frames=frames, blit=True, repeat=False)

        #anim.save(f"{key}_angle_animation.gif", fps=10)
        plt.show()
        plt.close(fig)
