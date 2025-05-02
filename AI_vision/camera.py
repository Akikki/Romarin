import cv2

def list_available_cameras(max_cameras=10):
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap is None or not cap.isOpened():
            # Could not open the camera, skip to next index
            continue
        else:
            available_cameras.append(index)
            cap.release()  # Release the camera once it's verified
    return available_cameras

if __name__ == "__main__":
    cameras = list_available_cameras(10)
    print("Available OpenCV cameras:", cameras)
