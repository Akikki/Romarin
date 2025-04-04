import cv2
import numpy as np
import imutils
import os
import serial
from pynput import keyboard

arduino = serial.Serial(port='COM6', baudrate=9600, timeout=1)

def send_command(mot):
    id, speed = (mot)
    if id in [1, 2, 3] and -255 <= speed <= 255:
        command = f"{id},{speed}\n"
        arduino.write(command.encode()) # Envoyer la commande encodée en bytes

def direc(x, y, x0, y0):
    mot1, mot2, mot3 = (1,0), (2,0), (3,0)
    dx=-(x-x0)
    dy=-(y-y0)
    
    if(dx<-10 or dx>10):
        mot1, mot2 = (1,round(dx*(255/1280))), (2, round(-dx*(255/1280)))
    if(dy<-10 or dy>10):
        mot3 = (3,round(dy*(255/480)))

    return (mot1, mot2, mot3)

def telecom(keys):
    z, q, s, d, c, v, Z, S = keys
    send_command((3, (c-v)*200))
    if Z==S==0:
        send_command((1, (z-s+q-d)*150))
        send_command((2, (z-s+d-q)*150))
    else:
        send_command((1, (Z-S+q-d)*250))
        send_command((2, (Z-S+d-q)*250))