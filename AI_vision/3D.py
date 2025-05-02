import cv2
import time
import threading
from ultralytics import YOLO
from pynput import keyboard
import tracking as tr

# ================================================
# ! Global Configuration and State Variables
# ================================================
# Enable sorting (for specific detection class)
sorting = False

# Manual control keys (updated by the keyboard listener)
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

# Start keyboard listener in the background\listening
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

# ================================================
# ! Initialize YOLO Model
# ================================================
model = YOLO("../models/best.pt")  # Load your YOLOv8 model
classNames = model.names

# ================================================
# ! Initialize OpenCV Camera
# ================================================
cap = cv2.VideoCapture(0)
# Set resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ================================================
# ! Detection and Caching Variables
# ================================================
skip_interval = 15
frame_count = 0
last_target = None
last_bbox = None
last_detection_time = 0
detection_timeout = 1.0  # seconds until detection is considered stale
running = True

# ================================================
# ! Motor Control Thread (Manual overrides Auto)
# ================================================
def motor_control_loop():
    global last_target, last_detection_time
    while running:
        if any(keys.values()):
            tr.telecom(list(keys.values()))
        else:
            if last_target is not None and (time.time() - last_detection_time) < detection_timeout:
                mot1, mot2, mot3 = tr.direc(last_target[0], last_target[1], 320, 240)
                tr.send_command(mot1)
                tr.send_command(mot2)
                tr.send_command(mot3)
            else:
                tr.send_command((1, 0))
                tr.send_command((2, 0))
                tr.send_command((3, 0))
        time.sleep(0.05)

motor_thread = threading.Thread(target=motor_control_loop, daemon=True)
motor_thread.start()

# ================================================
# ! Main Loop: Capture Frames, Run Detection, and Draw
# ================================================
try:
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frame_count += 1

        # Run detection every skip_interval frames
        if frame_count % skip_interval == 0:
            results = model(img, stream=True)
            detection_made = False
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if not sorting or classNames[cls] == "cell phone":
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = float(box.conf[0])
                        center_x = int((x1 + x2) // 2)
                        center_y = int((y1 + y2) // 2)
                        last_target = (center_x, center_y)
                        last_bbox = (int(x1), int(y1), int(x2), int(y2), classNames[cls], round(conf, 2))
                        last_detection_time = time.time()
                        detection_made = True
                        break
                if detection_made:
                    break

        # Clear stale detection
        if time.time() - last_detection_time > detection_timeout:
            last_target = None
            last_bbox = None

        # Draw bounding box
        if last_bbox is not None:
            x1, y1, x2, y2, class_name, conf = last_bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(img, center, 3, (0, 0, 255), -1)
            cv2.putText(img, f"{class_name} {conf}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('n'):
            break

finally:
    running = False
    motor_thread.join()
    cap.release()
    cv2.destroyAllWindows()
