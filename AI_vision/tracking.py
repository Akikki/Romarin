import cv2
import numpy as np
import os
import serial
from pynput import keyboard

arduino = serial.Serial(port='COM6', baudrate=9600, timeout=1)

def send_command(mot):
    id, speed = (mot)
    if id in [1, 2, 3] and -255 <= speed <= 255:
        command = f"{id},{speed}\n"
        arduino.write(command.encode()) # Envoyer la commande encodée en bytes

def direc(x, y, x0, y0, x1, y1, x2, y2):
    dx, dy = -(x-x0), -(y-y0)
    tx, ty = abs(x1-x2), abs(y1-y2)
    if (tx or ty) < 50:
        if(dx<-10 or dx>10):
            mot1, mot2 = (1, round(dx*(255/1280))), (2, round(-dx*(255/1280)))
        if(dy<-10 or dy>10):
            mot3 = (3,round(dy*(255/480)))
    else:
        if(dx<-10 or dx>10):
            mot1, mot2 = (1, round((dx*(255/1280)/2)+127)), (2, round((-dx*(255/1280))/2+127))
        if(dy<-10 or dy>10):
            mot3 = (3,round(dy*(255/480)))
    return (mot1, mot2, mot3)

def telecom(keys):
    z, q, s, d, c, v, Z, S = keys
    send_command((3, (c-v)*200))
    send_command((1, (z+(2*Z)-s-(2*S)+q-d)*125))
    send_command((2, (z+(2*Z)-s-(2*S)+d-q)*125))