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


# You can add more transforms here
transformations = {
    "rotate_15": rotate_15,
    "rotate_-15": rotate_minus15,
}

# === Process each video ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        continue

    input_path = os.path.join(input_folder, filename)
    print(f"Processing: {filename}")

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
        output_filename = f"{os.path.splitext(filename)[0]}_{name}.mp4"
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
