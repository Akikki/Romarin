import serial
import time

arduino = serial.Serial(port='COM6', baudrate=9600, timeout=1)

def unocom():
    # Remplacez 'COM3' par le port COM de votre Arduino

    send_command(1, 200)
    send_command(1, 100)
    send_command(1, 50)
    send_command(1, 200)


    arduino.close()
    print("\nCommunication terminée.")


def send_command(motor_id, speed):
    """Envoie une commande au format 'motor_id,speed' à l'Arduino."""
    if motor_id in [1, 2, 3] and 0 <= speed <= 255:
        command = f"{motor_id},{speed}\n"
        arduino.write(command.encode()) # Envoyer la commande encodée en bytes
        time.sleep(0.1) # Attendre un court instant pour laisser l'Arduino répondre
        response = arduino.readline().decode('utf-8').rstrip() # Lire la réponse
        print(f"Arduino: {response}")
    else:
        print("Commande invalide !")

if __name__=='__main__':
    unocom()
