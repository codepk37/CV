import cv2
import os
import glob

# Define parameters
image_folder = "anchor"
output_video = "anchor2.mp4"
frame_rate = 5  # Adjust as needed

# Get list of images sorted numerically (this ensures correct order)
image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

# Read the first image to get dimensions
first_frame = cv2.imread(image_files[0])
height, width, layers = first_frame.shape

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Write images to the video
for img_path in image_files:
    frame = cv2.imread(img_path)
    video.write(frame)

# Release resources
video.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_video}")
