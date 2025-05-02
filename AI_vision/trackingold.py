import cv2
import numpy as np
import imutils
import os
import serial

arduino = serial.Serial(port='COM6', baudrate=9600, timeout=1)

def send_command(mot):
    id, speed = (mot)
    if id in [1, 2, 3] and -255 <= speed <= 255:
        command = f"{id},{speed}\n"
        arduino.write(command.encode()) # Envoyer la commande encodée en bytes

def direc(x, y, x0, y0):
    dx=-(x-x0)
    dy=-(y-y0)

    match (dx, dy):
        case (dx, dy) if -10<=dx<=10 and -10<=dy<=10:   #centré
            return (1, 0), (2, 0), (3, 0)
        
        case (dx, dy) if -10<=dx<=10 and dy<-10:    #bas
            return (1, 0), (2, 0), (3, round(dy*(255/480)))
        case (dx, dy) if -10<=dx<=10 and dy>10: #haut
            return (1, 0), (2, 0), (3, round(dy*(255/480)))
        case (dx, dy) if dx<-10 and -10<=dy<=10:    #gauche
            return (1, round(dx*(255/1280))), (2,round(dx*(255/1280))), (3,0)
        case (dx, dy) if dx>10 and -10<=dy<=10: #droite
            return (1, round(dx*(255/1280))), (2,round(dx*(255/1280))), (3,0)

        case (dx, dy) if dx<-10 and dy<-10: #bas-gauche
            return (1, round(dx*(255/1280))), (2, round(dx*(255/1280))), (3, round(dy*(255/480)))
        case (dx, dy) if dx<-10 and dy>10:  #haut-gauche
            return (1, round(dx*(255/1280))), (2, round(dx*(255/1280))), (3, round(dy*(255/480)))
        case (dx, dy) if dx>10 and dy>10:   #haut-droite
            return (1, round(dx*(255/1280))), (2, round(dx*(255/1280))), (3, round(dy*(255/480)))
        case (dx, dy) if dx>10 and dy<-10:  #bas-droite
            return (1, round(dx*(255/1280))), (2, round(dx*(255/1280))), (3, round(dy*(255/480)))
        case _:
            return (1, 0), (2, 0), (3,0)    #si erreur