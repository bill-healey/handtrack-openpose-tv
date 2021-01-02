import cv2
import time
import threading


class Camera:
    last_frame = None
    last_ready = None

    def __init__(self, rtsp_link):
        vcap = cv2.VideoCapture(rtsp_link)
        thread = threading.Thread(target=self.rtsp_cam_buffer, args=(vcap,), name="rtsp_thread")
        thread.daemon = True
        thread.start()

    def rtsp_cam_buffer(self, vcap):
        frame_count = 0
        while True:
            if frame_count == 0:
                self.last_ready, self.last_frame = vcap.read()
            else:
                vcap.grab()
            frame_count = (frame_count + 1) % 2

    def getFrame(self):
        if (self.last_ready is not None) and (self.last_frame is not None):
            return self.last_frame.copy()
        else:
            return None


if __name__ == '__main__':
    cam = Camera(0)

    while cv2.waitKey(1) != 27:
        frame = cam.getFrame()
        if frame is not None:
            cv2.imshow('live', frame)
            time.sleep(0.1)
    cv2.destroyAllWindows()
