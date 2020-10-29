import cv2


class GestureTrainer:
    def __init__(self, gesture_name):
        self.gesture_name = gesture_name
        self.images_captured = 0
        self.capture_frequency_ms = 200

    def capture_training_image(self, frame, rectangle):
        cv2.imshow('Raw Training Image', frame)
        print(rectangle)
        # cropped_img = frame[x, y, w, h]
        #cv2.imshow('Cropped Training Image', cropped_img)
        # cv2.imwrite('{}_{}'.format(self.gesture_name, self.images_captured), crop_img)
        #self.images_captured += 1
