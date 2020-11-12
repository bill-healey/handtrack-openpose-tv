import cv2
import pyopenpose as op
import numpy as np

LEFT_HAND = 0
RIGHT_HAND = 1
BODY_MAP = {'Nose': 0, 'Neck': 1, 'RShoulder': 2, 'RElbow': 3, 'RWrist': 4, 'LShoulder': 5, 'LElbow': 6, 'LWrist': 7,
            'MidHip': 8, 'RHip': 9,
            'RKnee': 10, 'RAnkle': 11, 'LHip': 12, 'LKnee': 13, 'LAnkle': 14, 'REye': 15, 'LEye': 16, 'REar': 17,
            'LEar': 18, 'LBigToe': 19,
            'LSmallToe': 20, 'LHeel': 21, 'RBigToe': 22, 'RSmallToe': 23, 'RHeel': 24, 'Background': 25}
HAND_MAP = {'Wrist': 0,
            'Thumb1CMC': 1, 'Thumb2Knuckles': 2, 'Thumb3IP': 3, 'Thumb4FingerTip': 4,
            'Index1Knuckles': 5, 'Index2PIP': 6, 'Index3DIP': 7, 'Index4FingerTip': 8,
            'Middle1Knuckles': 9, 'Middle2PIP': 10, 'Middle3DIP': 11, 'Middle4FingerTip': 12,
            'Ring1Knuckles': 13, 'Ring2PIP': 14, 'Ring3DIP': 15, 'Ring4FingerTip': 16,
            'Pinky1Knuckles': 17, 'Pinky2PIP': 18, 'Pinky3DIP': 19, 'Pinky4FingerTip': 20}


class PoseTracker:

    def __init__(self):
        self.params = dict()
        self.params["model_folder"] = "D:\\git\\openpose\\models"
        self.params["hand"] = True
        self.params["face"] = False
        self.params["number_people_max"] = 1

        # Starting OpenPose
        self.open_pose_wrapper = op.WrapperPython()
        self.open_pose_wrapper.configure(self.params)
        self.open_pose_wrapper.start()

    def predict_pose_from_frame(self, frame, display_pose=True):
        datum = op.Datum()
        datum.cvInputData = frame
        self.open_pose_wrapper.emplaceAndPop([datum])

        pose_detected = datum.poseKeypoints.shape == (1, 25, 3)
        both_hands_detected = type(datum.handKeypoints) is list and len(datum.handKeypoints) == 2

        if pose_detected and both_hands_detected:
            pose = self._check_hand_pose(datum.poseKeypoints[0],
                                         datum.handKeypoints[0][0],
                                         datum.handKeypoints[1][0])

            pose['hand_rectangles'] = datum.handRectangles

            if display_pose:
                cv2.imshow("OpenPose Output", datum.cvOutputData)

            return pose
        else:
            return None

    def _check_hand_pose(self, body, left_hand, right_hand):
        pose_data = {
            'righthand_fingertip_coordinates': None,
            'lefthand_up': None,
            'righthand_up': None
        }

        if right_hand[HAND_MAP['Index4FingerTip']][1] == 0 or \
           right_hand[HAND_MAP['Wrist']][1] == 0 or \
           body[BODY_MAP['RShoulder']][1] == 0 or \
           left_hand[HAND_MAP['Index4FingerTip']][1] == 0:
            return pose_data

        shoulder_height = np.average((body[BODY_MAP['RShoulder']][1], body[BODY_MAP['LShoulder']][1]))
        right_hand_above_shoulder = body[BODY_MAP['RWrist']][1] < shoulder_height

        if right_hand_above_shoulder:
            pose_data['righthand_up'] = True
            pose_data['righthand_fingertip_coordinates'] = right_hand[HAND_MAP['Index4FingerTip']][0], \
                right_hand[HAND_MAP['Index4FingerTip']][1]
            pose_data['righthand_wrist_coordinates'] = body[BODY_MAP['RWrist']][0], body[BODY_MAP['RWrist']][1]

        if left_hand[HAND_MAP['Wrist']][1] < body[BODY_MAP['RShoulder']][1] and \
                left_hand[HAND_MAP['Index4FingerTip']][1] < body[BODY_MAP['LShoulder']][1] and \
                right_hand_above_shoulder:
            pose_data['lefthand_up'] = True
            print('Lefthand up')

        return pose_data

    def _print_datum_keypoints(self, datum):
        # Retrieve Mapping
        keypoint_map = self.open_pose_wrapper.getPoseBodyPartMapping(op.BODY_25)
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

