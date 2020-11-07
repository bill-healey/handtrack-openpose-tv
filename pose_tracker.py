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

    def check_hand_pose(self, body, left_hand, right_hand):
        # For now this is manual, will convert to detecting various poses via neural net model

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

        right_hand_above_shoulder = right_hand[HAND_MAP['Wrist']][1] < shoulder_height and \
           right_hand[HAND_MAP['Index4FingerTip']][1] < shoulder_height

        if right_hand_above_shoulder:
            pose_data['righthand_up'] = True
            pose_data['righthand_fingertip_coordinates'] = right_hand[HAND_MAP['Index4FingerTip']][0], \
                                                        right_hand[HAND_MAP['Index4FingerTip']][1]
            #print('RShoulder: {} Fingertip: {}'.format(shoulder_height, pose_data['righthand_fingertip_coordinates']))

        if left_hand[HAND_MAP['Wrist']][1] < body[BODY_MAP['RShoulder']][1] and \
                left_hand[HAND_MAP['Index4FingerTip']][1] < body[BODY_MAP['LShoulder']][1] and \
                right_hand_above_shoulder:
            pose_data['lefthand_up'] = True
            print('Lefthand up')

        return pose_data

    def get_pose_and_frame_from_file(self):
        lefthand_up_image = 'D:\\git\\openpose\\examples\\media\\COCO_val2014_000000000569.jpg'
        lefthand_down_image = 'D:\\git\\openpose\\examples\\media\\COCO_val2014_000000000241.jpg'
        hands_up_image = 'D:\\git\\openpose\\examples\\media\\hands_up.jpg'
        imageToProcess = cv2.imread(hands_up_image)
        pose = self.predict_pose_from_frame(imageToProcess, display_pose=True)

        return pose, imageToProcess

    def predict_pose_from_frame(self, frame, display_pose=True):
        self.datum.cvInputData = frame
        self.open_pose_wrapper.emplaceAndPop([self.datum])

        pose = None

        if self.datum.poseKeypoints.shape == (1, 25, 3) and \
                type(self.datum.handKeypoints) is list and \
                len( self.datum.handKeypoints) == 2:
            pose = self.check_hand_pose(self.datum.poseKeypoints[0],
                                        self.datum.handKeypoints[0][0],
                                        self.datum.handKeypoints[1][0])

            pose['hand_rectangles'] = self.datum.handRectangles

        if display_pose:
            cv2.imshow("OpenPose Output", self.datum.cvOutputData)

        return pose

    def print_all_keypoints(self):
        # Retrieve Mapping
        keypoint_map = self.open_pose_wrapper.getPoseBodyPartMapping(op.BODY_25)
        print({v: k for k, v in keypoint_map.items()})
        print(self.datum.handKeypoints[0][0])
        for index, keypoint in enumerate(self.datum.poseKeypoints[0]):
            print("{}: {}".format(keypoint_map[index], keypoint))

        print("\nLeft Hand")
        for i in range(len(HAND_MAP)):
             print("{}: {}".format(HAND_MAP[i], self.datum.handKeypoints[0][0][i]))

        print("\nRight Hand")
        for i in range(len(HAND_MAP)):
             print("{}: {}".format(HAND_MAP[i], self.datum.handKeypoints[1][0][i]))
        return

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
        self.datum = op.Datum()
