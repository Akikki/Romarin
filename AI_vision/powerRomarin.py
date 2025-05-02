from ultralytics import YOLO
import cv2
import time
import threading
import tracking as tr
from pynput import keyboard

# * Enable sorting
sorting = True
# * Mode de controle (control mode)
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

# * Use a smaller/faster model for detection
model = YOLO("yolo11n.pt")
classNames = model.names

# Start the keyboard listener in a background thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

# Initialize a single camera using OpenCV VideoCapture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Frame skip logic for YOLO detection
skip_interval = 15
frame_count = 0
last_boxes = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # The frame from VideoCapture is already in BGR color space
        img = frame

        # Check if any key in the keys dictionary is active
        if 1 in keys.values():
            tr.telecom(list(keys.values()))
        else:
            tr.telecom(list(keys.values()))
            frame_count += 1
            if frame_count % skip_interval == 0:
                results = model(img, stream=True)
                new_boxes = []
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        # Perform filtering based on class name if sorting is enabled
                        if not sorting or classNames[cls] == "cell phone":
                            x1, y1, x2, y2 = box.xyxy[0]
                            conf = float(box.conf[0])
                            new_boxes.append((x1, y1, x2, y2, conf, cls))
                last_boxes = new_boxes

            # Draw detection boxes on the image
            for (x1, y1, x2, y2, conf, cls) in last_boxes:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(img, (center_x, center_y), 3, (0, 0, 255), -1)
                confidence = round(conf, 2)
                class_name = classNames[cls]
                text = f"{class_name} {confidence}"
                cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                
                mot1, mot2, mot3 = tr.direc(center_x, center_y, 320, 240)
                # Optionally send commands multiple times based on the frame count and skip interval
                tr.send_command(mot1)
                tr.send_command(mot2)
                tr.send_command(mot3)

        # Display the detection window
        cv2.imshow("Detection", img)
        # Check if the 'n' key was pressed to break the loop
        if cv2.waitKey(1) == ord('n'):
            tr.send_command((1, 0))
            tr.send_command((2, 0))
            tr.send_command((3, 0))
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
