import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

# --- 1. INITIALIZATION ---

# Load YOLOv8 model (Phase 2)
print("Loading YOLO model...")
yolo_model = YOLO('models/yolov8n.pt') 

# Load MediaPipe Pose model (Phase 2)
print("Loading MediaPipe Pose model...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,       # We are processing a single image
    model_complexity=1,           # 0, 1, or 2. Higher = more accurate but slower.
    min_detection_confidence=0.5
)

# Define constants for key landmarks
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP
LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE
RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE

# --- 2. LOAD & PRE-PROCESS IMAGE ---

IMAGE_PATH = 'data/input_videos/batsman.jpg'
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Error: Could not load image from {IMAGE_PATH}")
    exit()

print(f"Image loaded successfully from {IMAGE_PATH}")
image_height, image_width, _ = image.shape

# --- 3. PHASE 2: DETECTION (YOLO) ---

# Run YOLO inference to find 'person' (class 0)
print("Running detection...")
results = yolo_model.predict(image, classes=[0], conf=0.6, verbose=False)

if len(results[0].boxes) == 0:
    print("No person (batsman) detected by YOLO.")
    exit()

# Get the bounding box of the first detected person
box = results[0].boxes[0]
x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]

print(f"Batsman detected at box: [{x1}, {y1}, {x2}, {y2}]")

# --- 4. PHASE 2: POSE ESTIMATION (MediaPipe) ---
# We run MediaPipe ONLY on the cropped region for efficiency

# Add padding to the box to ensure the whole body is included
padding = 30
crop_x1 = max(0, x1 - padding)
crop_y1 = max(0, y1 - padding)
crop_x2 = min(image_width, x2 + padding)
crop_y2 = min(image_height, y2 + padding)

# Crop the image
cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

if cropped_image.size == 0:
    print("Error: Cropped image is empty. Check bounding box logic.")
    exit()

# Convert the cropped image (BGR) to RGB for MediaPipe
cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

# Process the cropped image
print("Running pose estimation...")
results = pose.process(cropped_image_rgb)

# --- 5. PHASE 3: LANDMARK EXTRACTION (THE "METRIC ENGINE") ---
# The coordinates from MediaPipe are NORMALIZED (0.0 to 1.0)
# and are relative to the CROPPED image.
# We must convert them back to the ORIGINAL image's coordinate space.

if not results.pose_landmarks:
    print("MediaPipe could not detect pose in the cropped image.")
    exit()

print("\n--- Biomechanical Coordinates (relative to original image) ---")

landmarks = results.pose_landmarks.landmark
crop_height, crop_width, _ = cropped_image.shape

# Helper function to convert normalized/cropped coords to original image coords
def get_original_coords(landmark_id):
    lm = landmarks[landmark_id.value]
    
    # 1. Un-normalize (relative to crop)
    lm_x_crop = int(lm.x * crop_width)
    lm_y_crop = int(lm.y * crop_height)
    
    # 2. Add crop offset (relative to original image)
    lm_x_orig = lm_x_crop + crop_x1
    lm_y_orig = lm_y_crop + crop_y1
    
    return (lm_x_orig, lm_y_orig)

# Get the key coordinates we need
left_hip_coords = get_original_coords(LEFT_HIP)
right_hip_coords = get_original_coords(RIGHT_HIP)
left_ankle_coords = get_original_coords(LEFT_ANKLE)
right_ankle_coords = get_original_coords(RIGHT_ANKLE)

# Calculate the hip midpoint (our proxy for Center of Mass)
hip_midpoint_x = int((left_hip_coords[0] + right_hip_coords[0]) / 2)
hip_midpoint_y = int((left_hip_coords[1] + right_hip_coords[1]) / 2)
hip_midpoint = (hip_midpoint_x, hip_midpoint_y)

print(f"Left Hip:     {left_hip_coords}")
print(f"Right Hip:    {right_hip_coords}")
print(f"Left Ankle:   {left_ankle_coords}")
print(f"Right Ankle:  {right_ankle_coords}")
print(f"HIP MIDPOINT (CoM): {hip_midpoint}")
print("---------------------------------------------------------")

# --- 6. PHASE 4: VISUALIZATION ---
print("Visualizing and saving output...")
output_image = image.copy()

# A. Draw the YOLO box (blue)
cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# B. Draw the key landmarks (red)
cv2.circle(output_image, left_hip_coords, 8, (0, 0, 255), -1)
cv2.circle(output_image, right_hip_coords, 8, (0, 0, 255), -1)
cv2.circle(output_image, left_ankle_coords, 8, (0, 0, 255), -1)
cv2.circle(output_image, right_ankle_coords, 8, (0, 0, 255), -1)

# C. Draw the Hip Midpoint (yellow)
cv2.circle(output_image, hip_midpoint, 10, (0, 255, 255), -1)

# Save the final image
OUTPUT_PATH = 'data/output_videos/test_image_annotated.jpg'
cv2.imwrite(OUTPUT_PATH, output_image)

# Clean up
pose.close()

print(f"\nSuccess! Annotated image saved to: {OUTPUT_PATH}")
print("Check the image to verify the detection and landmarks.")