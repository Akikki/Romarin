import cv2  # OpenCV bibliothèque open source spécialisée dans le traitement d'images
import numpy as np  # simplifier certaines opérations courantes liées au traitement d'images
import imutils
import os

#detection formes
class ShapeDetector:
    def __init__(self):
        pass
 
    def detect(self, c):
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "carre" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagone"
        elif len(approx) == 6:
            shape = "hexagone"
        elif len(approx) == 10 or len(approx) == 12:
            shape = "etoile"
        else:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if circularity >= 0.7:
                shape = "Panneau STOP"
 
        return shape

#detection couleurs
def convert_rgb_to_names(rgb_tuple):
    r, g, b = rgb_tuple
 
    red_lower = (200, 0, 0)
    red_upper = (255, 100, 100)
    green_lower = (0, 200, 0)
    green_upper = (100, 255, 100)
    blue_lower = (0, 0, 200)
    blue_upper = (100, 100, 255)
 
    if red_lower[0] <= r <= red_upper[0] and red_lower[1] <= g <= red_upper[1] and red_lower[2] <= b <= red_upper[2]:
        return 'rouge'
    elif green_lower[0] <= r <= green_upper[0] and green_lower[1] <= g <= green_upper[1] and green_lower[2] <= b <= green_upper[2]:
        return 'vert'
    elif blue_lower[0] <= r <= blue_upper[0] and blue_lower[1] <= g <= blue_upper[1] and blue_lower[2] <= b <= blue_upper[2]:
        return 'bleu'
    else:
        max_component = max(r, g, b)
        if max_component == r:
            return 'rouge'
        elif max_component == g:
            return 'vert'
        else:
            return 'bleu'

#traitement d'image & tracking
def detect_shapes_and_print_results():
    sd = ShapeDetector()
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam
 
    while True:
        ret, frame = cap.read()
 
        if frame is None:
            print("Error: Could not read frame from the camera.")
            break
        #x, y, w, h = [int(v) for v in frame]
        resized = imutils.resize(frame, width=300)
        ratio = frame.shape[0] / float(resized.shape[0])
 
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
 
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
 
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)
 
                shape = sd.detect(c)
 
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
 
                mask = np.zeros(frame.shape[:2], np.uint8)
                cv2.drawContours(mask, [c], -1, 255, -1)
                imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mean = cv2.mean(imgRGB, mask=mask)[:3]
                named_color = convert_rgb_to_names(mean)
 
                mean2 = (255 - mean[0], 255 - mean[1], 255 - mean[2])
 
                objLbl = shape + " {}".format(named_color)
                textSize = cv2.getTextSize(objLbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.putText(frame, objLbl, (int(cX - textSize[0] / 2), int(cY + textSize[1] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            mean2, 2)
 
        cv2.imshow("Frame", frame)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    detect_shapes_and_print_results()