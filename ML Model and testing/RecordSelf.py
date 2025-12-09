import tkinter as tk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import torch
import torch.nn.functional as F
import pyro.distributions as dist

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

window_length = 10

model_data = np.load('lift_samples.npz', allow_pickle=True)
# Squat samples
squat_knee_samples = model_data['squat_knee'].item()
squat_hip_samples = model_data['squat_hip'].item()
squat_elbow_samples = model_data['squat_elbow'].item()
squat_shoulder_samples = model_data['squat_shoulder'].item()

# Deadlift samples
deadlift_knee_samples = model_data['deadlift_knee'].item()
deadlift_hip_samples = model_data['deadlift_hip'].item()
deadlift_elbow_samples = model_data['deadlift_elbow'].item()
deadlift_shoulder_samples = model_data['deadlift_shoulder'].item()

# Bench samples
bench_knee_samples = model_data['bench_knee'].item()
bench_hip_samples = model_data['bench_hip'].item()
bench_elbow_samples = model_data['bench_elbow'].item()
bench_shoulder_samples = model_data['bench_shoulder'].item()

print(squat_knee_samples['mu'])

# Joints to track
JOINTS_TO_TRACK = {
    "left_elbow": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
    "right_elbow": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
    "left_knee": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE],
    "right_knee": [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE],
    "left_hip": [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER],
    "right_hip": [mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER],
    "left_shoulder": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW],
    "right_shoulder": [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW],
}



class LiftRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lift Recorder")
        self.root.geometry("800x600")
        self.root.configure(bg='black')

        # Initialize MediaPipe vars
        self.cap = None
        self.pose = None
        self.running = False
        self.landmark_positions = {}
        self.angle_history = {}
        self.angle_windows = {}

        # Create selection UI
        self.create_selection_ui()

    def create_selection_ui(self):

        for widget in self.root.winfo_children():
            widget.destroy()

        # Title
        self.title_label = tk.Label(self.root, text="Select Lift Type", font=("Arial", 24))
        self.title_label.pack(pady=20)


        # Buttons
        lifts = ["Squat", "Bench", "Deadlift"]
        for lift in lifts:
            btn = tk.Button(self.root, text=lift, font=("Arial", 18), width=15, command=lambda l=lift: self.start_recording(l))
            btn.pack(pady=10)

    def start_recording(self, lift_type):
        self.lift_type = lift_type

        # Clear selection UI
        for widget in self.root.winfo_children():
            widget.destroy()

        # Expand window
        self.root.geometry("1200x1000")

        # Video label
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        # Stop button
        self.stop_button = tk.Button(self.root, text="STOP", font=("Arial", 24), bg="red", fg="white", command=self.stop)
        self.stop_button.pack(pady=10)

        # Initialize MediaPipe
        self.cap = cv2.VideoCapture(0)
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.running = True

        # Initialize data storage
        self.landmark_positions = {position.value: [[],[],[]] for position in mp_pose.PoseLandmark}
        self.angle_history = {joint: [] for joint in JOINTS_TO_TRACK}
        self.angle_windows = {joint: deque([], maxlen=window_length) for joint in JOINTS_TO_TRACK}

        # Start loop
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        frame = cv2.resize(frame, (960, 720))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks and angles
        try:
            # EXTRACT LANDMARKS AND APPEND X,Y,Z COORDS TO LANDMARK_POSITIONS
            landmarks = results.pose_landmarks.landmark

            for position in mp_pose.PoseLandmark:
                landmark = landmarks[position.value]
                self.landmark_positions[position.value][0].append(round(landmark.x, 2))
                self.landmark_positions[position.value][1].append(round(landmark.y, 2))
                self.landmark_positions[position.value][2].append(round(landmark.z, 2))

            # GO THRU EACH JOINT GIVEN IN JOINTS_TO_TRACK
            for joint_name, landmark_indices in JOINTS_TO_TRACK.items():

                try:  # EXECUTE IF JOINT HAS ACTIVE DATAq
                    a = landmarks[landmark_indices[0].value]
                    b = landmarks[landmark_indices[1].value]
                    c = landmarks[landmark_indices[2].value]

                    # Use only x, y for 2D analysis
                    angle = functions.calculate_angle(
                        [a.x, a.y, a.z],
                        [b.x, b.y, b.z],
                        [c.x, c.y, c.z]
                    )

                    # ADD CURRENT ANGLE AT FRAME TO ANGLE_WINDOWS. IF ANGLE_WINDOWS IS AS LONG AS THE DEQUE EARLIER,
                    # TAKE THE MEAN OF THE DEQUE. DEQUE IS UPDATED AT EACH FRAME, WITH THE FIRST VALUE IN THE DEQUE
                    # BEING REMOVED AS THE LAST IS ADDED (FIFO DATA STRUCTURE)
                    self.angle_windows[joint_name].append(angle)
                    if len(self.angle_windows[joint_name]) == window_length:
                        rolling_avg = np.mean(self.angle_windows[joint_name])
                        self.angle_history[joint_name].append(int(rolling_avg))


                # IF LANDMARK IS NOT FOUND IN FOOTAGE, SET ANGLE TO NOT A NUMBER
                except Exception:
                    angle = np.nan

                # IF WE HAVE AN ACTIVE ANGLE, DISPLAY THE ANGLE ON SCREEN
                if not np.isnan(angle):
                    b_coords = np.multiply([b.x, b.y], [image.shape[1], image.shape[0]]).astype(int)
                    cv2.putText(
                        image,
                        f"{joint_name}: {int(angle)}",
                        tuple(b_coords),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
            # IF WE CANNOT EXTRACT LANDMARKS, JUST PASS
        except:
            pass

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # Convert to Tkinter image
        img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        self.video_label.imgtk = img
        self.video_label.config(image=img)

        # Repeat loop
        self.root.after(10, self.update_frame)

    def stop(self):
        self.running = False
        if self.cap: self.cap.release()
        if self.pose: self.pose.close()
        self.video_label.config(image="")

        # Save data
        points = {f"landmark_{k}": np.array(v) for k,v in self.landmark_positions.items()}
        angles = {k: np.array(v) for k,v in self.angle_history.items()}
        functions.save_lift_data("recordSelf 1", points, angles, filename_tag="recordSelf")


        ################
        script_dir = os.path.dirname(os.path.abspath("../ML Model and Testing/RecordSelf.py"))
        data_dir = os.path.join(script_dir, '..', 'lift data')
        data = np.load(f"{data_dir}\\recordSelf files\\recordSelf 1 lift data.npz")

        RECORDSELF_TENSORS = {}

        tensor_dict = {}
        for key in data.files:

            lift_name_without_video_type = " ".join(key.split("_")[-2:])
            tensor_dict["recordself 1 good " + lift_name_without_video_type] = torch.tensor(data[key])

        RECORDSELF_TENSORS['recordself 1 good'] = tensor_dict
        recordself_joint_angles = functions.joint_angles_to_list(RECORDSELF_TENSORS)
        if self.lift_type == 'Squat':
            joint = 'knee'
        if self.lift_type == 'Deadlift':
            joint = 'hip'
        if self.lift_type == 'Bench':
            joint = 'elbow'

        recordself_split_reps = functions.split_lifts_into_reps(lift_name=self.lift_type.lower(), lift_list=recordself_joint_angles,
                                                                joint=joint)

        def resample(series: torch.Tensor, target_len: int = 60) -> torch.Tensor:
            """
            Linearly resamples a 1D tensor to target_len.
            Input:  shape (T,)
            Output: shape (target_len,)
            """
            series = series.unsqueeze(0).unsqueeze(0).to(torch.float32)  # (1,1,T)
            out = F.interpolate(series, size=target_len, mode='linear', align_corners=True)
            return out.squeeze()

        for repetition in range(len(recordself_split_reps)):
            for key in recordself_split_reps[repetition]['angles']:
                recordself_split_reps[repetition]['angles'][key] = resample(
                    recordself_split_reps[repetition]['angles'][key], target_len=60)
        ################



        self.show_plots(recordself_split_reps)

    def show_plots(self, split_reps):
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.geometry("1200x800")
        self.root.configure(bg="#2e2e2e")  # dark gray

        tk.Label(self.root, text=f"{self.lift_type} Joint Angles", font=("Arial", 24)).pack(pady=10)

        # Since we only have a single rep at a time
        rep = split_reps[0]  # take first rep
        angles = rep['angles']

        # Plot angles
        fig, axes = plt.subplots(4, 2, figsize=(12, 11))
        fig.patch.set_facecolor('black')
        axes = axes.flatten()


        for i, (joint, data) in enumerate(angles.items()):
            ax = axes[i]
            if isinstance(data, torch.Tensor):
                data = data.numpy()

            ### NEXT STEP joint is not in model_data, only knee, hip, etc.. are and joint is left_knee, left_hip, etc...
            ### ALSO, split reps doesn't seem to be working
            ax.set_facecolor("#1c1c1c")  #
            ax.title.set_color("white")  # title color
            ax.xaxis.label.set_color("white")  # x label color
            ax.yaxis.label.set_color("white")  # y label color
            ax.tick_params(axis='x', colors='white')  # x-axis ticks white
            ax.tick_params(axis='y', colors='white')  # y-axis ticks white

            if self.lift_type.lower() + '_' + joint.split(' ')[1] in model_data:
                print(self.lift_type.lower() + '_' + joint.split(' ')[1])
                samples = model_data[self.lift_type.lower() + '_' + joint.split(' ')[1]].item()  # expected dict with 'mu' and 'sigma'
                mu_samples = samples['mu']
                sigma_samples = samples['sigma']
                num_draws, num_frames = mu_samples.shape

                # Posterior mean & CI
                mu_mean = mu_samples.mean(dim=0).numpy()
                lower_q = (1 - .9) / 2.0
                upper_q = 1 - lower_q
                mu_lower = mu_samples.quantile(lower_q, dim=0).numpy()
                mu_upper = mu_samples.quantile(upper_q, dim=0).numpy()

                # Posterior predictive
                y_pred = dist.Normal(mu_samples, sigma_samples).sample().numpy()
                ypred_mean = y_pred.mean(axis=0)
                ypred_lower = np.quantile(y_pred, lower_q, axis=0)
                ypred_upper = np.quantile(y_pred, upper_q, axis=0)

                frames = np.arange(num_frames)

                # Plot posterior mean & CI
                ax.plot(frames, ypred_mean, color='#ff7070', lw=3, label='posterior predictive mean')
                ax.fill_between(frames, ypred_lower, ypred_upper, alpha=0.25, color='#76fc72',
                                label=f'{int(.9 * 100)}% posterior predictive')

            ax.plot(data, color='#7ee3fc', lw=3, label='your rep')

            if i < 6:  # first 6 plots in 4x2 grid are top 3 rows
                ax.set_xticks([])
            else:
                ax.set_xlabel("Frame", fontsize=10, color='white')
            ax.set_title(joint, fontsize=13)
            if i % 2 == 0:
                ax.set_ylabel("Angle", fontsize=10)
            ax.set_ylim(0, 185)
            ax.tick_params(axis='both', which='major', labelsize=8)
            if i == 0:
                ax.legend(fontsize=10)






        # Embed figure in Tkinter
        canvas_fig = FigureCanvasTkAgg(fig, master=self.root)
        canvas_fig.draw()
        canvas_fig.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Back button
        tk.Button(self.root, text="Back to Lift Selection", font=("Arial", 18),
                  command=self.create_selection_ui).pack(pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.bind("<Escape>", lambda e: root.destroy())  # press ESC to close window

    app = LiftRecorderApp(root)
    root.mainloop()


#NPZ VIEWING
# import numpy as np
# import matplotlib.pyplot as plt
#
# data_dir = os.path.join(script_dir, '..', 'lift data')
# data = np.load(f"{data_dir}\\recordSelf files\\recordSelf angles.npz")
#
#
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