from ultralytics import YOLO
import cv2
import math
import tracking as tr
import threading
from pynput import keyboard
import pdb
import time

# mode de controle
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

# Démarrer l'écouteur clavier en arrière-plan
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("yolo11s.pt")
# model = YOLO("best.pt")

# object classes
classNames = model.names

while True:
    if 1 in keys.values():
            tr.telecom(list(keys.values()))
    
    else:
        tr.telecom(list(keys.values()))
        success, img = cap.read()
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                cls = int(box.cls[0])
                if classNames[cls] == "cell phone":
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                    # draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                    # compute and draw center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    cv2.circle(img, (center_x, center_y), radius=3, color=(0, 0, 255), thickness=-1)

                    mot1, mot2, mot3 = tr.direc(center_x, center_y, 320, 240)
                    tr.send_command(mot1)
                    tr.send_command(mot2)
                    tr.send_command(mot3)

                    # confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    #print("Confidence --->", confidence)

                    # class name
                    #print("Class name -->", classNames[cls])

                    # object details text
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 255, 255)
                    thickness = 2
                    cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('n'):
        tr.send_command((1, 0))
        tr.send_command((2, 0))
        tr.send_command((3, 0))
        break

cap.release()
cv2.destroyAllWindows()
