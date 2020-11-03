import cv2
import time
from pytv import PyTVCursor
from pose_tracker import PoseTracker
from gesture_trainer import GestureTrainer

cursor = PyTVCursor()
cam = cv2.VideoCapture(0)
pose_tracker = PoseTracker()
gesture_trainer = GestureTrainer()

while cv2.waitKey(1) != 27:
    ret, frame = cam.read()
    pose = pose_tracker.get_pose_from_frame(frame)

    if pose is None:
        print("No pose found")
        time.sleep(1)
        continue

    cursor.update_world_coordinate(pose['righthand_fingertip_coordinates'])
    if pose['lefthand_up']:
        if gesture_trainer:
            gesture_trainer.capture_training_image(frame, pose['hand_rectangles'])
        else:
            cursor.click()

cam.release()
cv2.destroyAllWindows()
