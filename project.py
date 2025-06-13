from calibration import a, b
# Now you can use a and b directly!
# ...rest of your code...
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
from ultralytics import YOLO
import cv2
import requests  # Add this import at the top
import pandas as pd  # Add this import for Excel export

model = YOLO(r"C:\Users\lenovo\Desktop\omar\best.onnx", task="detect")
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Unable to access the camera")
    exit()

# Keep reading frames for 2 seconds to keep the stream alive
start_time = time.time()
frame = None
while time.time() - start_time < 2:
    ret, frame = cap.read()
    if not ret:
        continue

# Now capture the frame you want
ret, frame = cap.read()
if not ret or frame is None:
    print("Failed to grab frame")
    cap.release()
    exit()

if frame.shape[2] == 4:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

results = model.predict(source=frame, conf=0.7, device='cpu')

# Parameters for real-world conversion
focal_length = 800  # in pixels
real_distance = 26.5  # in mm
#a =  21.563342318 # scaling factor (set this to your value)
#b =  2.315843887   # scaling factor (set this to your value)



def send_firebase_command(command: str):
    url = "https://djin-78525-default-rtdb.firebaseio.com/GRIPPER/Command.json"
    response = requests.put(url, json=command)
    if response.ok:
        print(f"Command '{command}' sent to Firebase.")
    else:
        print(f"Failed to send command: {response.text}")

box_data = []  # To store box info for Excel

# Draw only the bounding boxes, and show width (X) and height (Y) as numbers (no "mm")
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw a green point at the center of the box
    center_x = x1 + (x2 - x1) // 2
    center_y = y1 + (y2 - y1) // 2
    cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)

    # --- Projection lines from center to axes ---
    height, width = frame.shape[:2]
    axis_x = width // 2
    axis_y = height // 2

    # Draw projection to X axis (horizontal center line)
    cv2.line(frame, (center_x, center_y), (center_x, axis_y), (0, 255, 0), 1)
    # Draw projection to Y axis (vertical center line)
    cv2.line(frame, (center_x, center_y), (axis_x, center_y), (0, 255, 0), 1)

    # Calculate signed pixel distances from center to axes
    dx_px = center_x - axis_x  # right is positive, left is negative
    dy_px = axis_y - center_y  # above is positive, below is negative

    # Convert to mm using your calibration
    dx_mm = (dx_px * real_distance) / focal_length * a 
    dy_mm = (dy_px * real_distance) / focal_length * b

    # Draw the signed distances on the projections (in green, bold)
    text_dx = f"{dx_mm:.1f}"
    text_dy = f"{dy_mm:.1f}"
    cv2.putText(frame, text_dx, (center_x + 5, axis_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, text_dy, (axis_x + 5, center_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Calculate width and height in pixels
    box_width_px = x2 - x1
    box_height_px = y2 - y1

    # Convert to mm using scaling factor
    box_width_mm = (box_width_px * real_distance) / focal_length * a / 10
    box_height_mm = (box_height_px * real_distance) / focal_length * b 

    # --- Add this to collect box data ---
    box_name = "boxname"  # Replace with actual name if available
    box_data.append({
        "Box Name": box_name,
        "Length": round(box_height_mm, 2),
        "Width": round(box_width_mm, 2),
        "Hauteur": ""  # This will be empty for now
    })

    # Send command to Firebase: "z(width_mm),(height_mm)"
    command_str = f"z{box_width_mm*20:.1f}"
    send_firebase_command(command_str)

    # Center of the box in pixels
    center_x = x1 + (x2 - x1) // 2
    center_y = y1 + (y2 - y1) // 2

    print(f"Box: longueur={box_height_mm:.2f}, largeur={box_width_mm:.2f}, center=({text_dx}, {text_dy})")
    print(f"{command_str}")  # This will print z... after the Box: ... line

    # Draw width at the top inside the box (centered, bright violet)
    text_w = f"w={box_width_mm:.1f}"
    text_w_size = cv2.getTextSize(text_w, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0]
    text_w_x = x1 + (box_width_px - text_w_size[0]) // 2
    text_w_y = y1 + text_w_size[1] + 5
    cv2.putText(frame, text_w, (text_w_x, text_w_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)

    # Draw height at the left inside the box (centered vertically, bright violet)
    text_h = f"h={box_height_mm:.1f}"
    text_h_size = cv2.getTextSize(text_h, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0]
    text_h_x = x1 + 5
    text_h_y = y1 + (box_height_px + text_h_size[1]) // 2
    cv2.putText(frame, text_h, (text_h_x, text_h_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)

# Draw X and Y axes (centered, no labels, no arrows)
height, width = frame.shape[:2]
center_x = width // 2
center_y = height // 2

cv2.line(frame, (0, center_y), (width, center_y), (255, 255, 255), 2)      # X axis in white
cv2.line(frame, (center_x, height), (center_x, 0), (255, 255, 255), 2)    # Y axis in white

# Draw 16 strips (15 lines) vertically and horizontally
num_strips = 16
height, width = frame.shape[:2]

# Vertical lines
for i in range(1, num_strips):
    x = int(i * width / num_strips)
    cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)

# Horizontal lines
for i in range(1, num_strips):
    y = int(i * height / num_strips)
    cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)

# Draw more (and slimmer) grid lines for finer graduation
num_strips = 32  # or higher for more graduations
height, width = frame.shape[:2]

# Use a very light gray for subtle, slim lines
slim_color = (220, 220, 220)  # RGB for light gray

# Vertical lines
for i in range(1, num_strips):
    x = int(i * width / num_strips)
    cv2.line(frame, (x, 0), (x, height), slim_color, 1)  # thickness=1

# Horizontal lines
for i in range(1, num_strips):
    y = int(i * height / num_strips)
    cv2.line(frame, (0, y), (width, y), slim_color, 1)  # thickness=1

output_path = r"C:\Users\lenovo\Desktop\omar\annotated_photo.jpg"
cv2.imwrite(output_path, frame)
print(f"Photo saved to {output_path}")

# --- Add this to save to Excel ---
import os
import pandas as pd

excel_path = r"C:\Users\lenovo\Desktop\omar\box_results.xlsx"

# Check if the Excel file exists and read it if it does
if os.path.exists(excel_path):
    old_df = pd.read_excel(excel_path)
    df = pd.concat([old_df, pd.DataFrame(box_data)], ignore_index=True)
else:
    df = pd.DataFrame(box_data)

df.to_excel(excel_path, index=False)
print(f"Excel saved to {excel_path}")

cap.release()
cv2.destroyAllWindows()