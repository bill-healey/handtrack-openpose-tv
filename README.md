## TV Gesture Control
Provides control of WebOS-compatible TVs without any type of controller by tracking human body arm and hand position via a consumer-grade webcam.  This project leverages the CMU Perceptual Computing Lab's excellent OpenPose library in addition to PyWebOSTV.

## Requirements
- [x] pywebostv==0.8.4
- [x] OpenPose
- [x] OpenCV

## OpenPose Windows Build Commands
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