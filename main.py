import cv2
import time
from pytv import PyTVCursor
from pose_tracker import PoseTracker
from gesture_trainer import GestureTrainer
from hysteresis import Hysteresis

cursor = PyTVCursor()
cam = cv2.VideoCapture(0)
pose_tracker = PoseTracker()
gesture_trainer = GestureTrainer()
hysteresis = Hysteresis()
image_collect_only = False
use_live_camera = True

while cv2.waitKey(1) != 27:
    if use_live_camera:
        ret, frame = cam.read()
        pose = pose_tracker.predict_pose_from_frame(frame)
    else:
        pose, frame = pose_tracker.get_pose_and_frame_from_file()

    if pose is None:
        time.sleep(0.5)
        continue

    if image_collect_only and pose['righthand_up']:
        gesture_trainer.capture_training_image(frame, pose['hand_rectangles'], 'rh', 'unrecognized')
        continue

    if pose['righthand_up']:
        gesture, confidence = gesture_trainer.predict_gesture_from_frame(frame, pose['hand_rectangles'])

        if confidence > 0.80:
            hysteresis.update_state(gesture)

        if hysteresis.is_stable('rh_animalhead', consecutive=1):
            cursor.update_world_coordinate(pose['righthand_wrist_coordinates'])
        if hysteresis.is_stable('rh_twoleft', secs=0.2, consecutive=2) and pose['lefthand_up']:
            cursor.keypad_back()
            hysteresis.reset()
            print('***Back***')
        if hysteresis.is_stable('rh_two', secs=0.2, consecutive=2) and pose['lefthand_up']:
            cursor.keypad_home()
            hysteresis.reset()
            print('***Home***')
        if hysteresis.is_stable('rh_two', secs=0.2, consecutive=2) and not pose['lefthand_up']:
            cursor.keypad_up()
            hysteresis.reset()
            print('***Up***')
        if hysteresis.is_stable('rh_twodown', secs=0.2, consecutive=2):
            cursor.keypad_down()
            hysteresis.reset()
            print('***Down***')
        if hysteresis.is_stable('rh_twoleft', secs=0.2, consecutive=2) and not pose['lefthand_up']:
            cursor.keypad_left()
            hysteresis.reset()
            print('***Left***')
        if hysteresis.is_stable('rh_tworight', secs=0.2, consecutive=2):
            cursor.keypad_right()
            hysteresis.reset()
            print('***Right***')
        if hysteresis.is_stable('rh_fingerspread', secs=0.2, consecutive=2):
            cursor.click()
            cursor.keypad_ok()
            hysteresis.reset()
            print('***Click***')
        if hysteresis.is_stable('rh_stop', secs=0.6, consecutive=3):
            cursor.toggle_pause()
            hysteresis.reset()
            print('***Pause***')

cam.release()
cv2.destroyAllWindows()
