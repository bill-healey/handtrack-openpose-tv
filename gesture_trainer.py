import os
import random
import shutil

import cv2
import time
import glob
import numpy as np
import tensorflow as tf
import keras
import datetime
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D


class GestureTrainer:
    def __init__(self):
        self.capture_frequency_sec = 0.2
        self.last_capture_timestamp = 0
        self.min_validation_images = 10
        self.min_test_images = 3
        self.gestures = [
            'unrecognized',
            'thumbsup',
            'thumbsdown',
            'thumbsleft',
            'thumbsright',
            'indexup',
            'indexleft',
            'indexright',
            'indexdown',
            'fist',
            'fingerspread',
            'spock',
            'two',
            'l',
            'animalhead',
            'stop',
            'surfer',
            'zero'
        ]
        self.gesture_image_files = self.scan_image_files()
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(physical_devices))
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def capture_training_image(self, frame, hand_rectangle_list, hand, gesture):
        for hands in hand_rectangle_list:
            if len(hands) != 2:
                return
            left_hand_rect = hands[0]
            right_hand_rect = hands[1]
        cur_time = time.time()
        if cur_time < self.last_capture_timestamp + self.capture_frequency_sec:
            return
        print('Frame shape {}'.format(frame.shape))
        if int(right_hand_rect.x) < 0 or int(right_hand_rect.x + right_hand_rect.width) > frame.shape[1] or \
           int(right_hand_rect.y) < 0 or int(right_hand_rect.y + right_hand_rect.height) > frame.shape[0]:
            return

        #cv2.rectangle(frame,
        #              (int(right_hand_rect.x), int(right_hand_rect.y)),
        #              (int(right_hand_rect.x + right_hand_rect.width), int(right_hand_rect.y + right_hand_rect.height)),
        #              (255, 0, 0),
        #              1)

        #cv2.imshow('Raw Training Image', frame)
        cropped_img = frame[int(right_hand_rect.y):int(right_hand_rect.y+right_hand_rect.height),
                            int(right_hand_rect.x):int(right_hand_rect.x+right_hand_rect.width)]
        cv2.imshow('Cropped Training Image', cropped_img)
        image_name = 'images/{}_{}_{}.jpg'.format(hand, gesture, self.gesture_image_files[hand][gesture]['next_int'])
        cv2.imwrite(image_name, cropped_img)
        print('Captured training image {}'.format(image_name))
        self.last_capture_timestamp = cur_time
        self.gesture_image_files[hand][gesture]['next_int'] += 1

    def scan_image_files(self):
        gesture_image_files = {}
        for hand in ['rh', 'lh']:
            gesture_image_files[hand] = {}
            for pose in self.gestures:
                gesture_image_files[hand][pose] = {
                    'train_filenames': glob.glob('images/{}_{}_*.jpg'.format(hand, pose), recursive=False),
                    'val_filenames': glob.glob('images/validation/{}_{}_*.jpg'.format(hand, pose), recursive=False),
                    'test_filenames': glob.glob('images/test/{}_{}_*.jpg'.format(hand, pose), recursive=False),
                }

                gesture_image_files[hand][pose]['all_filenames'] = gesture_image_files[hand][pose]['train_filenames'] + \
                    gesture_image_files[hand][pose]['val_filenames'] + \
                    gesture_image_files[hand][pose]['test_filenames']

                if len(gesture_image_files[hand][pose]['all_filenames']) == 0:
                    gesture_image_files[hand][pose]['next_int'] = 0
                else:
                    gesture_image_files[hand][pose]['next_int'] = max(
                        [int(filename.split('_')[-1].split('.')[0]) for filename in gesture_image_files[hand][pose]['all_filenames']]) + 1

                print('Found {} images ({} train, {} validation, {} test) for pose {}_{} next int {}'.format(
                    len(gesture_image_files[hand][pose]['all_filenames']),
                    len(gesture_image_files[hand][pose]['train_filenames']),
                    len(gesture_image_files[hand][pose]['val_filenames']),
                    len(gesture_image_files[hand][pose]['test_filenames']),
                    hand,
                    pose,
                    gesture_image_files[hand][pose]['next_int']
                ))

        return gesture_image_files

    def rebalance_test_train_files(self):
        rescan = False
        for hand, poses in self.gesture_image_files.items():
            for pose, groups in poses.items():
                if len(groups['train_filenames']) <= self.min_validation_images:
                    print('Insufficient training files to create validation images for pose {}-{}'.format(hand, pose))
                    continue

                # Check for sufficient validation images
                if len(groups['val_filenames']) < self.min_validation_images:
                    new_validation_images = random.sample(groups['train_filenames'],
                                                          self.min_validation_images - len(groups['val_filenames']))
                    for file in new_validation_images:
                        shutil.move(file, 'images/validation/{}'.format(os.path.basename(file)))
                        rescan = True
        if rescan:
            self.gesture_image_files = self.scan_image_files()

        rescan = False
        for hand, poses in self.gesture_image_files.items():
            for pose, groups in poses.items():
                if len(groups['train_filenames']) <= self.min_validation_images + self.min_test_images:
                    print('Insufficient training files to create test images for pose {}-{}'.format(hand, pose))
                    continue

                # Check for sufficient test images
                if len(groups['test_filenames']) < self.min_test_images:
                    new_test_images = random.sample(groups['train_filenames'],
                                                          self.min_test_images - len(groups['test_filenames']))
                    for file in new_test_images:
                        shutil.move(file, 'images/test/{}'.format(os.path.basename(file)))
                        rescan = True

        if rescan:
            self.gesture_image_files = self.scan_image_files()


    def load_training_images(self, hands, poses):
        x = {
            'train': [],
            'val': [],
            'test': []
        }
        y = {
            'train': [],
            'val': [],
            'test': []
        }
        for hand in hands:
            for pose in poses:
                for t in ['train', 'val', 'test']:
                    for file in self.gesture_image_files[hand][pose][f'{t}_filenames']:
                        img = keras.preprocessing.image.load_img(file)
                        # print('{} {} {} {}'.format(type(img), img.format, img.mode, img.size))
                        img_array = keras.preprocessing.image.img_to_array(img)
                        img_array = keras.preprocessing.image.smart_resize(img_array, (80, 80), interpolation='bilinear')
                        x[t].append(img_array)
                        y[t].append('{}_{}'.format(hand, pose))

        datagen = ImageDataGenerator(
            preprocessing_function=keras.applications.vgg16.preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='nearest'
        )

        for t in ['train', 'val', 'test']:
            x[t] = np.array(x[t])
            y[t] = np.array(y[t])

        datagen.fit(x['train'])

        for x_batch, y_batch in datagen.flow(x['train'], y['train'], batch_size=9):
            for i in range(0, len(x_batch)):
                pyplot.subplot(330 + 1 + i)
                pyplot.imshow(x_batch[i])
            pyplot.show()

    # def train(self):
    #     batch_size - 128
    #     epocs = 10
    #     learning_rate = 0.01
    #
    #     model_name = 'hand-pose-right-{}-{}.h5'.format(datetime.date.today(), epocs)
    #     #model_name = 'hand-pose-left-{}-{}.h5'.format(datetime.date.today(), epocs)
    #
    #     # Model Definition
    #     model = Sequential()
    #     model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    #     model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Dropout(rate=0.25))
    #     model.add(Flatten())
    #     model.add(Dense(units=128, activation='relu'))
    #     model.add(Dropout(rate=0.5))
    #     model.add(Dense(units=num_classes, activation='softmax'))
    #     model.compile(loss=keras.losses.categorical_crossentropy,
    #                   optimizer=keras.optimizers.Adam(lr=learning_rate),
    #                   metrics=['accuracy'])


if __name__ == '__main__':
    trainer = GestureTrainer()
    trainer.rebalance_test_train_files()
    trainer.load_training_images(['rh'], ['unrecognized'])

