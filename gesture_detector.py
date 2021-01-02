import datetime
import glob
import os
import random
import shutil
import time
from hashlib import md5

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot
from sklearn.metrics import classification_report, confusion_matrix

class GestureDetector:
    def __init__(self, prediction_model_filename=None):
        self.capture_frequency_sec = 0.2
        self.last_capture_timestamp = 0
        self.min_validation_images = 50
        self.min_test_images = 20
        self.gestures = [
            'animalhead',
            'fingerspread',
            'fist',
            'indexdown',
            'indexleft',
            'indexright',
            'indexup',
            'l',
            'spock',
            'stop',
            'surfer',
            'thumbsdown',
            'thumbsleft',
            'thumbsright',
            'thumbsup',
            'two',
            'unrecognized',
            'zero',
            'twoleft',
            'tworight',
            'twodown',
        ]
        self.gesture_image_files = None
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        print("GPUs Available: ", len(physical_devices))
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        try:
            self.prediction_model = keras.models.load_model(prediction_model_filename)
            self.prediction_decoder = np.load(f'{prediction_model_filename}.npy')
        except Exception as e:
            self.prediction_model = None
            print('Could not load prediction model from file {} due to {}'.format(prediction_model_filename, e))

    def predict_gesture_from_frame(self,
                                   frame,
                                   people_hand_rectangles,
                                   capture_low_confidence_training_img=False,
                                   capture_high_confidence_training_img=False):
        cropped_hand_img = self.extract_right_hand_from_images(frame, people_hand_rectangles)
        if cropped_hand_img is None:
            return None, 0.0
        img_array = keras.preprocessing.image.img_to_array(cropped_hand_img).astype('float32') / 255
        img_array = keras.preprocessing.image.smart_resize(img_array, (40, 40), interpolation='bilinear')
        img_array = np.expand_dims(img_array, axis=0)  # Make this single image a rank 4 tensor
        predictions = self.prediction_model.predict(img_array)[0]
        pred_i = np.argmax(predictions)
        if predictions[pred_i] > 0.80:
            print('high confidence gesture: {}, confidence: {}'.format(self.prediction_decoder[pred_i],
                                                                       predictions[pred_i]))
            if capture_high_confidence_training_img:
                self.capture_training_image(frame,
                                            people_hand_rectangles,
                                            *self.prediction_decoder[pred_i].split('_'),
                                            subdir='autocap_high_confidence')
        else:
            print('low confidence gesture: {}, confidence: {}'.format(self.prediction_decoder[pred_i], predictions[pred_i]))
            if capture_low_confidence_training_img:
                self.capture_training_image(frame,
                                            people_hand_rectangles,
                                            *self.prediction_decoder[pred_i].split('_'),
                                            subdir='autocap_low_confidence')

        return self.prediction_decoder[pred_i], predictions[pred_i]

    def extract_right_hand_from_images(self, frame, people_hand_rectangles):
        for person_hands in people_hand_rectangles:
            if len(person_hands) != 2:
                return None
            left_hand_rect = person_hands[0]
            right_hand_rect = person_hands[1]

        if right_hand_rect.x < 0.:
            right_hand_rect.x = 0.

        if int(right_hand_rect.x) < 0 or int(right_hand_rect.x + right_hand_rect.width) > frame.shape[1] or \
                int(right_hand_rect.y) < 0 or int(right_hand_rect.y + right_hand_rect.height) > frame.shape[0]:
            return None

        cropped_img = frame[int(right_hand_rect.y):int(right_hand_rect.y+right_hand_rect.height),
                      int(right_hand_rect.x):int(right_hand_rect.x+right_hand_rect.width)]

        return cropped_img

    def capture_training_image(self, frame, people_hand_rectangles, hand, gesture, subdir='autocaptured'):
        cur_time = time.time()

        if cur_time < self.last_capture_timestamp + self.capture_frequency_sec:
            return

        cropped_img = self.extract_right_hand_from_images(frame, people_hand_rectangles)

        if cropped_img is not None:
            cv2.imshow('Cropped Training Image', cropped_img)
            image_name = 'images/{}/{}_{}_{}.jpg'.format(
                subdir,
                hand,
                gesture,
                md5(np.ascontiguousarray(cropped_img)).hexdigest()[:6])
            cv2.imwrite(image_name, cropped_img)
            print('Captured training image {}'.format(image_name))
            self.last_capture_timestamp = cur_time

    def scan_image_files(self):
        gesture_image_files = {}
        for hand in ['rh']:
            gesture_image_files[hand] = {}
            for pose in self.gestures:
                gesture_image_files[hand][pose] = {
                    'train_filenames': glob.glob('images/{}_{}_*.jpg'.format(hand, pose), recursive=False),
                    'val_filenames': glob.glob('images/validation/{}_{}_*.jpg'.format(hand, pose), recursive=False),
                    'test_filenames': glob.glob('images/test/{}_{}_*.jpg'.format(hand, pose), recursive=False),
                    'autocaptured_filenames': glob.glob('images/autocaptured/{}_{}_*.jpg'.format(hand, pose), recursive=False),
                }

                # This takes any files matching the pattern, hashes it, and renames using the convention.
                bad_filenames = glob.glob('images/{}_{} (*).jpg'.format(hand, pose), recursive=False)
                for file in bad_filenames:
                    img = keras.preprocessing.image.load_img(file, color_mode='rgb')
                    try:
                        os.rename(file, 'images\\{}_{}_{}.jpg'.format(hand, pose, md5(np.ascontiguousarray(img)).hexdigest()[:6]))
                    except FileExistsError:
                        print('Deleting {} since it is a duplicate'.format(file))
                        os.remove(file)

                gesture_image_files[hand][pose]['all_filenames'] = gesture_image_files[hand][pose]['train_filenames'] + \
                                                                   gesture_image_files[hand][pose]['val_filenames'] + \
                    gesture_image_files[hand][pose]['test_filenames'] + \
                    gesture_image_files[hand][pose]['autocaptured_filenames']

                print('Found {} images ({} train, {} validation, {} test) for pose {}_{}'.format(
                    len(gesture_image_files[hand][pose]['all_filenames']),
                    len(gesture_image_files[hand][pose]['train_filenames']),
                    len(gesture_image_files[hand][pose]['val_filenames']),
                    len(gesture_image_files[hand][pose]['test_filenames']),
                    hand,
                    pose
                ))

        return gesture_image_files

    def rebalance_test_train_files(self):
        self.gesture_image_files = self.scan_image_files()
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

    def train(self, hands, poses, show_training_images=False):
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
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        if self.gesture_image_files is None:
            self.gesture_image_files = self.scan_image_files()

        for hand in hands:
            for pose in poses:
                for t in ['train', 'val', 'test']:
                    for file in self.gesture_image_files[hand][pose][f'{t}_filenames']:
                        img = keras.preprocessing.image.load_img(file, color_mode='rgb')
                        img_array = keras.preprocessing.image.img_to_array(img)
                        img_array = keras.preprocessing.image.smart_resize(img_array, (40, 40), interpolation='bilinear')
                        x[t].append(img_array)
                        y[t].append('{}_{}'.format(hand, pose))

        train_datagen = ImageDataGenerator(
            preprocessing_function=keras.applications.mobilenet.preprocess_input,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='nearest',
            #rescale=1./255
        )

        val_datagen = ImageDataGenerator(
            preprocessing_function=keras.applications.mobilenet.preprocess_input,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='nearest',
            #rescale=1./255
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=keras.applications.mobilenet.preprocess_input,
            #rescale=1./255
        )

        for t in ['train', 'val', 'test']:
            x[t] = np.array(x[t])
            y[t] = np.array(y[t]).reshape(-1, 1)
            y[t] = enc.fit_transform(y[t])  # Encode Y using OneHot

        train_datagen.fit(x['train'], augment=True)
        val_datagen.fit(x['val'], augment=True)
        test_datagen.fit(x['test'])

        if show_training_images:
            for x_batch, y_batch in train_datagen.flow(x['train'], y['train'], batch_size=100):
                for i in range(0, len(x_batch)):
                    subplot = pyplot.subplot(10, 10, i + 1)
                    subplot.set_title(enc.inverse_transform(y_batch[i].reshape(1, -1))[0][0])
                    pyplot.imshow(x_batch[i])
                pyplot.subplots_adjust(left=0, right=1.0, bottom=0.025, top=0.975, wspace=0.155, hspace=0.470)
                pyplot.get_current_fig_manager().window.showMaximized()
                pyplot.show()
                break

        print('About to train with {} training images'.format(len(x['train'])))

        learning_rate = 0.001
        epocs = 70
        batch_size = 14
        model_name = 'hand-pose-right-{}-{}.h5'.format(datetime.date.today(), epocs)

        train_gen = train_datagen.flow(x['train'], y['train'], batch_size=batch_size, shuffle=True)
        val_gen = val_datagen.flow(x['val'], y['val'], batch_size=batch_size, shuffle=True)
        test_gen = test_datagen.flow(x['test'], y['test'], batch_size=batch_size, shuffle=False)

        mobile_net = tf.keras.applications.mobilenet.MobileNet()
        sixth_to_last = mobile_net.layers[-6].output
        output = Dense(units=len(enc.categories_[0]), activation='softmax')(sixth_to_last)
        model = tf.keras.Model(inputs=mobile_net.input, outputs=output)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])

        print("Training on {} training images, {} validation images model {}".format(
            train_gen.n,
            val_gen.n,
            model_name))

        model.fit(train_gen,
                  epochs=epocs,
                  steps_per_epoch=train_gen.n // train_gen.batch_size,
                  validation_data=val_gen,
                  verbose=2)

        # Save the model and the one-hot encoding so we can decode later
        model.save(model_name)
        np.save(f'{model_name}', enc.categories_[0])

        predictions = model.predict(test_gen, verbose=2)
        y_pred = np.argmax(predictions, axis=1)
        cm_labels = np.argmax(test_gen.y, axis=1)
        cm = confusion_matrix(test_gen.classes, y_pred).numpy()
        cm_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        cm_df = pd.DataFrame(cm_norm,
                             index=enc.categories_[0],
                             columns=enc.categories_[0])
        figure = plt.figure(figsize=(8, 8))
        sns.heatmap(cm_df, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


if __name__ == '__main__':
    trainer = GestureDetector()
    trainer.rebalance_test_train_files()
    gestures_to_train = [
        'animalhead',
        'fingerspread',
        'fist',
        'stop',
        'two',
        'twoleft',
        'tworight',
        'twodown',
    ]
    trainer.train(['rh'], gestures_to_train, show_training_images=True)

