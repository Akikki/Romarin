from ultralytics import YOLO
import cv2
import time
from picamera2 import Picamera2
import threading

# Use a smaller/faster model for the Pi
model = YOLO("yolov8n.pt")
classNames = model.names

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

stop_event = threading.Event()

# Shared variables and their locks
main_frame = None
main_lock = threading.Lock()

detection_frame = None
detection_lock = threading.Lock()

second_frame = None
second_lock = threading.Lock()

def capture_main_camera():
    global main_frame
    while not stop_event.is_set():
        frame_rgb = picam2_main.capture_array()
        img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        with main_lock:
            main_frame = img

def run_detection():
    """Run YOLO detection on the latest main camera frame without blocking UI updates."""
    global detection_frame
    skip_interval = 5
    frame_count = 0
    while not stop_event.is_set():
        with main_lock:
            # Get a copy of the latest main frame
            frame_copy = main_frame.copy() if main_frame is not None else None
        if frame_copy is None:
            continue

        frame_count += 1
        # Only run detection every 'skip_interval' frames
        if frame_count % skip_interval == 0:
            results = model(frame_copy, stream=True)
            boxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    boxes.append((x1, y1, x2, y2, conf, cls))
            # Draw bounding boxes on the frame copy
            for (x1, y1, x2, y2, conf, cls) in boxes:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame_copy, (center_x, center_y), 3, (0, 0, 255), -1)
                confidence = round(conf, 2)
                class_name = classNames[cls]
                text = f"{class_name} {confidence}"
                cv2.putText(frame_copy, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
        with detection_lock:
            detection_frame = frame_copy

def capture_second_camera():
    global second_frame
    while not stop_event.is_set():
        frame_rgb2 = picam2_second.capture_array()
        img2 = cv2.cvtColor(frame_rgb2, cv2.COLOR_RGB2BGR)
        with second_lock:
            second_frame = img2

# Start background threads for capturing and processing
main_cap_thread = threading.Thread(target=capture_main_camera, daemon=True)
detect_thread = threading.Thread(target=run_detection, daemon=True)
second_cap_thread = threading.Thread(target=capture_second_camera, daemon=True)

main_cap_thread.start()
detect_thread.start()
second_cap_thread.start()

# Main UI loop: update both windows as fast as possible
try:
    while not stop_event.is_set():
        with detection_lock:
            disp_detection = detection_frame.copy() if detection_frame is not None else None
        with second_lock:
            disp_second = second_frame.copy() if second_frame is not None else None

        if disp_detection is not None:
            cv2.imshow("Picamera Detection", disp_detection)
        if disp_second is not None:
            cv2.imshow("Second Camera", disp_second)

        # One waitKey to handle both windows; press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
finally:
    picam2_main.stop()
    picam2_second.stop()
    cv2.destroyAllWindows()
