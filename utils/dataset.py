import os
import cv2

from constants import SAVE_INTERVAL


gesture_name = "one"  # change between recordings
save_dir = f"data/raw/{gesture_name}"
os.makedirs(save_dir, exist_ok=True)


def handle_save_image(cropped_bin_filled):
    global frame_count
    frame_count += 1
    if frame_count % SAVE_INTERVAL == 0:
        filename = os.path.join(save_dir, f"{(frame_count):04d}.png")
        cv2.imwrite(filename, cropped_bin_filled)
        print(f"Saved {filename}")
