# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import pyopenpose as op
import numpy as np
from pytv import PyTVCursor

LEFT_HAND = 0
RIGHT_HAND = 1

BODY_MAP = {'Nose': 0, 'Neck': 1, 'RShoulder': 2, 'RElbow': 3, 'RWrist': 4, 'LShoulder': 5, 'LElbow': 6, 'LWrist': 7, 'MidHip': 8, 'RHip': 9,
            'RKnee': 10, 'RAnkle': 11, 'LHip': 12, 'LKnee': 13, 'LAnkle': 14, 'REye': 15, 'LEye': 16, 'REar': 17, 'LEar': 18, 'LBigToe': 19,
            'LSmallToe': 20, 'LHeel': 21, 'RBigToe': 22, 'RSmallToe': 23, 'RHeel': 24, 'Background': 25}

HAND_MAP = {'Wrist': 0,
            'Thumb1CMC': 1, 'Thumb2Knuckles': 2, 'Thumb3IP': 3, 'Thumb4FingerTip': 4,
            'Index1Knuckles': 5, 'Index2PIP': 6, 'Index3DIP': 7, 'Index4FingerTip': 8,
            'Middle1Knuckles': 9, 'Middle2PIP': 10, 'Middle3DIP': 11, 'Middle4FingerTip': 12,
            'Ring1Knuckles': 13, 'Ring2PIP': 14, 'Ring3DIP': 15, 'Ring4FingerTip': 16,
            'Pinky1Knuckles': 17, 'Pinky2PIP': 18, 'Pinky3DIP': 19, 'Pinky4FingerTip': 20}


def check_hand_pose(body, left_hand, right_hand):
    # Right hand wrist above shoulders is used as cursor activation

    # print('RWrist: {}'.format(body[BODY_MAP['RWrist']]))
    # print('REye: {}'.format(body[BODY_MAP['REye']]))
    retval = {
        'finger_pos': None,
        'click': None
    }

    if right_hand[HAND_MAP['Index4FingerTip']][1] == 0 or \
       right_hand[HAND_MAP['Wrist']][1] == 0 or \
       body[BODY_MAP['RShoulder']][1] == 0 or \
       left_hand[HAND_MAP['Index4FingerTip']][1] == 0:
        return retval

    shoulder_height = np.average((body[BODY_MAP['RShoulder']][1], body[BODY_MAP['LShoulder']][1]))

    if right_hand[HAND_MAP['Wrist']][1] < shoulder_height and \
       right_hand[HAND_MAP['Index4FingerTip']][1] < shoulder_height:

        retval['finger_pos'] = right_hand[HAND_MAP['Index4FingerTip']][0], right_hand[HAND_MAP['Index4FingerTip']][1]
        print('RShoulder: {} Fingertip: {}'.format(shoulder_height, retval['finger_pos']))

    # Left hand wrist above shoulders triggers a click
    if left_hand[HAND_MAP['Wrist']][1] < body[BODY_MAP['RShoulder']][1] and \
            left_hand[HAND_MAP['Index4FingerTip']][1] < body[BODY_MAP['LShoulder']][1]:
        retval['click'] = True
        print('Click')

    return retval


def configure_for_images(parser):

    # Read image and face rectangle locations
    lefthand_up_image = 'D:\\git\\openpose\\examples\\media\\COCO_val2014_000000000569.jpg'
    lefthand_down_image = 'D:\\git\\openpose\\examples\\media\\COCO_val2014_000000000241.jpg'
    imageToProcess = cv2.imread(lefthand_up_image)

    # Create new datum
    datum = op.Datum()
    datum.cvInputData = imageToProcess

    # Process and display image
    opWrapper.emplaceAndPop([datum])


def print_all_keypoints(op, datum):
    # Retrieve Mapping
    keypoint_map = op.getPoseBodyPartMapping(op.BODY_25)
    print({v: k for k, v in keypoint_map.items()})
    print(datum.handKeypoints[0][0])
    for index, keypoint in enumerate(datum.poseKeypoints[0]):
        print("{}: {}".format(keypoint_map[index], keypoint))

    print("\nLeft Hand")
    for i in range(len(HAND_MAP)):
         print("{}: {}".format(HAND_MAP[i], datum.handKeypoints[0][0][i]))

    print("\nRight Hand")
    for i in range(len(HAND_MAP)):
         print("{}: {}".format(HAND_MAP[i], datum.handKeypoints[1][0][i]))
    return


cam = cv2.VideoCapture(0)
#try:
# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "D:\\git\\openpose\\models"
params["hand"] = True
params["face"] = False
params["number_people_max"] = 1

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()
cursor = PyTVCursor()

while cv2.waitKey(1) != 27:
    ret, frame = cam.read()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    if datum.poseKeypoints.shape == (1, 25, 3) and type(datum.handKeypoints) is list and len(datum.handKeypoints) == 2:
        pose = check_hand_pose(datum.poseKeypoints[0],
                               datum.handKeypoints[0][0],
                               datum.handKeypoints[1][0])
        cursor.update_world_coordinate(pose['finger_pos'])
        if pose['click']:
            cursor.click()
    else:
        print("No pose found")
        time.sleep(1)

    cv2.imshow("OpenPose 1.6.0 Output", datum.cvOutputData)

#except Exception as e:
#    print(e)
#    raise e
#    sys.exit(-1)

#finally:
#    cam.release()
#    cv2.destroyAllWindows()
