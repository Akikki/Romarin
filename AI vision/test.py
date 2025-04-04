from pynput import keyboard
import time

def on_press(key):
    try:
        print(key)
    except AttributeError:
        pass

def on_release(key):
    try:
        print('000')
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

while True:
    
    time.sleep(0.1)
