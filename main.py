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
    pose = pose_tracker.predict_pose_from_frame(frame)
    #pose, frame = pose_tracker.get_pose_and_frame_from_file()

    if pose is None:
        #print("No pose found")
        time.sleep(1)
        continue

    cursor.update_world_coordinate(pose['righthand_fingertip_coordinates'])
    if pose['righthand_up']:
        #gesture_trainer.capture_training_image(frame, pose['hand_rectangles'], 'rh', 'unrecognized')
        gesture_trainer.predict_gesture_from_frame(frame, pose['hand_rectangles'])
    #if pose['lefthand_up']:
    #    cursor.click()

cam.release()
cv2.destroyAllWindows()
