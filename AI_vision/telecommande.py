import math
import tracking as tr
import threading
from pynput import keyboard

# mode de controle
keys = {'z': 0, 'q': 0, 's': 0, 'd': 0, 'c': 0, 'v': 0, 'Z': 0, 'S': 0}

def on_press(key):
    try:
        if key.char in keys:
            keys[key.char] = 1
            print(keys.get(key.char))
        elif key in keys:
            keys[key] = 1
            print(keys.get(key))
    except AttributeError:
        pass

def on_release(key):
    try:
        if key == keyboard.Key.esc:
            return False
        elif key.char in keys:
            keys[key.char] = 0
    except AttributeError:
        if key == keyboard.Key.shift:
            keys['maj'] = 0

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

while True:
    if 1 in keys:
        print('g')
        tr.telecom(keys.values())