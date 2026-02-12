import cv2


class camera:
    def __init__(self, path: int = 0):
        self.path = path

    def start_stream(self):
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at {self.path}")

    def get_frame(self):
        frame = self.cap.read()
        if not frame == None:
            raise RuntimeError(f"Failed to open camera at {self.path}")
        return frame

    def set_config(self, height, width, fps):
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera stream not started. Call start_stream() first")

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        except Exception as e:
            print(f"Error setting camera config: {e}")
            return False

        return True
