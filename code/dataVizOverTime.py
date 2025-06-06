import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = np.load(r"C:\Users\alexk\PycharmProjects\SeniorProject\data\deadlift files\deadlift angles.npz")

for key in data:
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

    # Save as GIF (no need for ImageMagick if pillow writer is default)
    anim.save(f"{key}_angle_animation.gif", fps=10)

    plt.close(fig)
