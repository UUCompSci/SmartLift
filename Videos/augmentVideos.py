import cv2
import os

input_folder = os.getcwd()
output_folder = os.getcwd()

def rotate_15(frame):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    return cv2.warpAffine(frame, M, (w, h))


def rotate_minus15(frame):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -15, 1.0)
    return cv2.warpAffine(frame, M, (w, h))

def mirror_video(frame):
    return cv2.flip(frame, 1)

# You can add more transforms here
transformations = {
    "rotate_15": rotate_15,
    "rotate_-15": rotate_minus15,
    "flip": mirror_video
}

total_bench_videos = 0
total_squat_videos = 0
total_deadlift_videos = 0
file_names = set()

# === Process each video ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        continue

    input_path = os.path.join(input_folder, filename)
    file_names.add(filename.split('.')[0])
    if filename.split(' ')[0] == 'bench':
        total_bench_videos += 1
    if filename.split(' ')[0] == 'squat':
        total_squat_videos += 1
    if filename.split(' ')[0] == 'deadlift':
        total_deadlift_videos += 1

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        continue

    input_path = os.path.join(input_folder, filename)
    print(f"Processing: {filename}")
    print(filename.split(".")[0])
    file_names.add(filename.split(".")[0])

    for name, transform in transformations.items():
        cap = cv2.VideoCapture(input_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_width, out_height = width, height

        # Try to determine new dimensions after transform (not required but better)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read first frame of {filename}")
            break
        transformed_frame = transform(frame)
        out_height, out_width = transformed_frame.shape[:2]

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
        print(filename.split('.')[0], "file_names=",file_names)


        lift_type = filename.split(' ')[1]
        filename = filename.split(' ')
        print(filename)
        print("lift_type=",lift_type)
        if filename[0] == 'bench':
            total_bench_videos += 1
            filename[1] = str(total_bench_videos)
        if filename[0] == 'squat':
            total_squat_videos += 1
            filename[1] = str(total_squat_videos)
        if filename[0] == 'deadlift':
            total_deadlift_videos += 1
            filename[1] = str(total_deadlift_videos)
        print(filename)
        filename = " ".join(filename)
        print('new filename is ',filename)


        output_filename = f"{os.path.splitext(filename)[0]}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            transformed = transform(frame)
            out.write(transformed)

        cap.release()
        out.release()
        print(f"Saved: {output_filename}")

print("✅ All videos processed.")
