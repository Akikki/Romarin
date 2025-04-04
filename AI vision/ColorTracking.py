# Importing all modules
import cv2
import numpy as np
import time
import threading
import queue
import tracking as tr


def hsv(hsv):
    h, s, v = hsv
    opencv_h = int((h / 360) * 179)
    opencv_s = int((s / 100) * 255)
    opencv_v = int((v / 100) * 255)

    return [opencv_h, opencv_s, opencv_v]


# Specifying upper and lower ranges of color to detect in hsv format
lower = np.array(hsv((300, 40, 10)))
upper = np.array(hsv((360, 90, 100)))

#lower = np.array([65, 100, 20])
#upper = np.array([80, 255, 255])

lower2 = np.array([170, 150, 20])
upper2 = np.array([179, 255, 255])

# Capturing webcam footage
webcam_video = cv2.VideoCapture(0)
data_queue = queue.Queue()


def print_periodically():
    while True:
        if not data_queue.empty():
            message = data_queue.get()  # Get the latest message
            print("Received:", message)
        time.sleep(0.1)  # Ensures it prints every 0.5 seconds


thread = threading.Thread(target=print_periodically, daemon=True)
thread.start()

rect_type = 1
count = 0
while True:
    success, video = webcam_video.read()  # Reading webcam footage

    img = cv2.cvtColor(video, cv2.COLOR_BGR2HSV)  # Converting BGR image to HSV format

    mask = cv2.inRange(img, lower, upper)  # Masking the image to find our color
    mask += cv2.inRange(img, lower2, upper2) # Adding other mask

    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)  # Finding contours in mask image

    # Finding position of all contours
    cv2.circle(video, (0, 0), 50, (255, 0, 0), -1)
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 200 and rect_type == 1:
                x, y, w, h = cv2.boundingRect(mask_contour)  # Getting the coordinate

                rect = cv2.minAreaRect(mask_contour)
                box = cv2.boxPoints(rect)
                box = np.int64(box)
                cv2.drawContours(video, [box], 0, (0, 0, 255), 4)
                c_x = x + w // 2  # center in x
                c_y = y + h // 2
                cv2.circle(video, (c_x, c_y), 5, (0, 0, 255), -1)  # Red dot at the center
                count += 1

                mot1, mot2, mot3 = tr.direc(c_x, c_y, 320, 240)
                tr.send_command(mot1)
                tr.send_command(mot2)
                tr.send_command(mot3)

                if count % 10 == 0:
                    print(c_x-320, c_y-240)

                # data_queue.put((c_x, c_y))

    cv2.imshow("mask image", mask)  # Displaying mask image

    cv2.imshow("window", video)  # Displaying webcam image

    if cv2.waitKey(1) & 0xFF == ord('q'):
        tr.send_command((1, 0))
        tr.send_command((2, 0))
        tr.send_command((3, 0))
        break

"""        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 300 and rect_type == 0:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(video, (x, y), (x + w, y + h), (0, 0, 255), 3)  #drawing rectangle
                c_x = x + w // 2  # center in x
                c_y = y + h // 2
                cv2.circle(video, (c_x, c_y), 5, (0, 0, 255), -1)  # Red dot at the center

                data_queue.put((c_x, c_y))"""
