from ultralytics import YOLO
import cv2
import time
from picamera2 import Picamera2
import threading
import tracking as tr
from pynput import keyboard
import pdb
import time

# * mode de controle
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

# * Use a smaller/faster model for the Pi
model = YOLO("../models/best.pt")
classNames = model.names

# Démarrer l'écouteur clavier en arrière-plan
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

# Initialize main camera (camera 0) for detection
picam2_main = Picamera2(camera_num=0)
config_main = picam2_main.create_preview_configuration(main={"size": (640, 480)})
picam2_main.configure(config_main)
picam2_main.start()
time.sleep(1)

# Initialize second camera (camera 1) for raw display
picam2_second = Picamera2(camera_num=1)
config_second = picam2_second.create_preview_configuration(main={"size": (640, 480)})
picam2_second.configure(config_second)
picam2_second.start()
time.sleep(1)

# Frame skip logic for YOLO detection
skip_interval = 5
frame_count = 0
last_boxes = []

# Shared event to signal shutdown
stop_event = threading.Event()
# Shared variable and lock to hold the latest frame from the second camera
second_frame = None
second_frame_lock = threading.Lock()

def update_second_camera():
    """ Continuously capture frames from camera 1 and store the latest frame. """
    global second_frame
    while not stop_event.is_set():
        frame_rgb2 = picam2_second.capture_array()
        img2 = cv2.cvtColor(frame_rgb2, cv2.COLOR_RGB2BGR)
        with second_frame_lock:
            second_frame = img2

# Start the background thread for updating the second camera frame
thread2 = threading.Thread(target=update_second_camera, daemon=True)
thread2.start()

try:
    while not stop_event.is_set():
        # Capture frame from main camera and process detection
        frame_rgb = picam2_main.capture_array()
        img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        if 1 in keys.values():
            tr.telecom(list(keys.values()))
        else:
            tr.telecom(list(keys.values()))
            success, img = cap.read()
            results = model(img, stream=True)

            frame_count += 1
            if frame_count % skip_interval == 0:
                results = model(img, stream=True)
                new_boxes = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        new_boxes.append((x1, y1, x2, y2, conf, cls))
                last_boxes = new_boxes

            # Draw detection boxes on the main image
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
                tr.send_command(mot1)
                tr.send_command(mot2)
                tr.send_command(mot3)

            # Display both windows in the main thread
            cv2.imshow("Picamera Detection", img)
            with second_frame_lock:
                if second_frame is not None:
                    cv2.imshow("Second Camera", second_frame)

            # Use one waitKey call to handle both windows; "q" sets the stop_event
            if cv2.waitKey(1) == ord('n'):
                tr.send_command((1, 0))
                tr.send_command((2, 0))
                tr.send_command((3, 0))
                break

finally:
    picam2_main.stop()
    picam2_second.stop()
    cv2.destroyAllWindows()
