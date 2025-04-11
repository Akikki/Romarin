from ultralytics import YOLO
import cv2
import time
import threading
import tracking as tr
from pynput import keyboard

# ================================================
# Global Configuration and State Variables
# ================================================
# Enable sorting if you want to filter detections (e.g., by object class)
sorting = True

# Manual control keys. Their values will be updated by the keyboard listener.
keys = {'z': 0, 'q': 0, 's': 0, 'd': 0, 'c': 0, 'v': 0, 'Z': 0, 'S': 0}

def on_press(key):
    try:
        if key.char in keys:
            keys[key.char] = 1
    except AttributeError:
        pass

def on_release(key):
    try:
        if key.char in keys:
            keys[key.char] = 0
    except AttributeError:
        pass

# Start keyboard listener (runs in its own thread)
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

# ================================================
# Initialize Model and Camera
# ================================================
model = YOLO("yolo11n.pt")
classNames = model.names

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Frequency to update detection (YOLO inference every N frames)
skip_interval = 10
frame_count = 0

# Variables to cache detection results
# last_target: center of the detected bounding box (x, y)
# last_bbox: full bounding box info (x1, y1, x2, y2, class_name, confidence) for drawing
last_target = None
last_bbox = None
last_detection_time = 0
# If detection is stale (no new detection within this many seconds), clear the target
detection_timeout = 1.0

# A flag for the threads; when set to False the threads will end
running = True

# ================================================
# Motor Control Thread (Manual Overrides Auto)
# ================================================
def motor_control_loop():
    """
    This loop runs independently (at roughly 20Hz). It checks for manual commands
    via the keys dictionary. If any manual key is active, it sends manual commands
    using tr.telecom. Otherwise, it sends auto-generated commands based on the last 
    detection location (last_target), provided that detection is recent.
    If the detection is stale, it sends stop commands.
    """
    global last_target, last_detection_time
    while running:
        if any(keys.values()):
            # Highest priority: if any manual key is pressed, send manual commands.
            # The telecom function should be designed to continuously drive the motors
            # based on the manual key state.
            tr.telecom(list(keys.values()))
        else:
            # No manual keys, so use the last detection if it's fresh.
            if last_target is not None and (time.time() - last_detection_time) < detection_timeout:
                mot1, mot2, mot3 = tr.direc(last_target[0], last_target[1], 320, 240)
                tr.send_command(mot1)
                tr.send_command(mot2)
                tr.send_command(mot3)
            else:
                # No detection (or too old); send stop commands.
                tr.send_command((1, 0))
                tr.send_command((2, 0))
                tr.send_command((3, 0))
        # Tune this sleep interval for motor control frequency (here, ~20 Hz)
        time.sleep(0.05)

# Start the motor control thread
motor_thread = threading.Thread(target=motor_control_loop, daemon=True)
motor_thread.start()

# ================================================
# Main Loop: Capture Frames, Run Detection, and Draw
# ================================================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Make a copy for drawing purposes.
        img = frame.copy()
        frame_count += 1

        # Run detection every 'skip_interval' frames.
        if frame_count % skip_interval == 0:
            results = model(img, stream=True)
            detection_made = False
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    # Filter detections if sorting is enabled (for example: only "cell phone")
                    if not sorting or classNames[cls] == "cell phone":
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = float(box.conf[0])
                        center_x = int((x1 + x2) // 2)
                        center_y = int((y1 + y2) // 2)
                        # Update cached target and drawing info
                        last_target = (center_x, center_y)
                        last_bbox = (int(x1), int(y1), int(x2), int(y2), classNames[cls], round(conf, 2))
                        last_detection_time = time.time()
                        detection_made = True
                        # Use the first valid detection (or implement a selection method if needed)
                        break
                if detection_made:
                    break

        # If the last detection is too old, clear cached values (auto-stop will occur in motor thread)
        if time.time() - last_detection_time > detection_timeout:
            last_target = None
            last_bbox = None

        # Draw the cached bounding box if available
        if last_bbox is not None:
            x1, y1, x2, y2, class_name, conf = last_bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(img, center, 3, (0, 0, 255), -1)
            cv2.putText(img, f"{class_name} {conf}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show the video feed with drawing
        cv2.imshow("Detection", img)
        if cv2.waitKey(1) == ord('n'):
            break

finally:
    running = False          # Signal threads to finish
    motor_thread.join()      # Wait for motor thread to close
    cap.release()
    cv2.destroyAllWindows()
