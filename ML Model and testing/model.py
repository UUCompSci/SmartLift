import tkinter as tk
import subprocess
import sys

def run_lift(lift_type):
    # Spawn new Python process running recordself.py
    subprocess.Popen([sys.executable, "recordself.py", lift_type])

root = tk.Tk()
root.title("Select Lift")

tk.Label(root, text="Choose a lift to record:").pack(pady=10)

btn_squat = tk.Button(root, text="Squat", command=lambda: run_lift("squat"))
btn_bench = tk.Button(root, text="Bench", command=lambda: run_lift("bench"))
btn_deadlift = tk.Button(root, text="Deadlift", command=lambda: run_lift("deadlift"))

btn_squat.pack(pady=5)
btn_bench.pack(pady=5)
btn_deadlift.pack(pady=5)

root.mainloop()
