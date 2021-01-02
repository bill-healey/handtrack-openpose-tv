## TV Gesture Control
Provides control of WebOS-compatible TVs without any type of controller by tracking human body arm and hand 
position via a consumer-grade webcam.  This project leverages the CMU Perceptual Computing Lab's excellent OpenPose 
library in addition to PyWebOSTV.

Run with:
```
python main.py
```

## Gestures

Most gestures are performed with the right-hand only.  

- **Cursor Movement**: Hold index and thumb together and cursor will follow movement of the right hand.
- **Keypad Directional Movement**: Two fingers spread in any direction (up, down, left, right)
- **Cursor Click / Keypad OK**: Open right hand wide with fingers spread
- **Play/Pause**: Operates as a toggle, hold right hand up in 'stop' position (fingers straight up and together)
- **Back**: Raise left hand, then signal keypad left with right hand (two fingers spread pointing left)
- **Menu**: Raise left hand, then signal keypad up with right hand (two fingers spread pointing up, aka peace sign)

## Implementation
OpenPose is used to detect the presence of a human along with the position of the hands.

The y-coordinate of the wrist vs. shoulders is used to determine whether the human has a hand raised.  
If a hand is raised, then then cropped images of the hands are passed into the GestureDetector module and a neural network
is used to determine the gesture that each hand is making.

The Hysteresis module is used as a filter to ensure that no action is taken until gestures meet a cofidence and stability threshold.

The PyTV module handles sending commands to the WebOSTV as well as mapping the world coordinate system onto a TV-friendly cursor coordinate system.  
This cursor mapping is implemented as a draggable window of configurable size within the world (webcam) coordinate system.

GestureDetector can be run standalone to capture training images, scan for and standardize the names of training images, and train a new version of the neural network model. 
GestureDetector also has the option of capturing low and or high confidence images while performing predictions in order to improve the accuracy of the model. 
Note that capturing images during prediction does create some lag and can make gesture detection less accurate.

## TODO:
- [x] The WebOSTv interface centers the cursor at times.  Cursor tracking logic within pytv.py needs to match that behavior.
- [x] Figure out how to package this is a more user-friendly way
- [x] Upload hand image data to Kaggle
- [x] Try temporal recurrence

## Requirements
- [x] pywebostv==0.8.4
- [x] OpenPose
- [x] OpenCV

## OpenPose Windows Build Steps
- [x] Install MSVC Native Tools for VS 2019
- [x] Install CUDA
- [x] Install cuDNN

- [x] Start VS2019 command prompt
- [x] Edit CMakeLists to turn on Python compilation
```
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
cd ../../models/
getModels.bat
mkdir build && cd build
cmake ..
msbuild /p:Configuration=Release ALL.vcxproj
msbuild /p:Configuration=Release INSTALL.vcxproj
```

- [x] Copy bin folder into Release folder
- [x] Add Release folder to windows env path variable
- [x] Copy python/openpose folder into Python's site-packages dir