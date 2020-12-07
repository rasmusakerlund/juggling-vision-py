# Juggling Vision python version

The project provides real-time localization of balls and hands in video as well as live categorization of some common juggling patterns. The code was written as part of my thesis project. If you want to understand the code you might want to have a look at the implementation chapter: http://www.diva-portal.org/smash/record.jsf?pid=diva2:1297966

## Demo Video
[![juggling vision demo](https://img.youtube.com/vi/nsTGB06gu40/0.jpg)](https://www.youtube.com/watch?v=nsTGB06gu40)

## Getting Started (tested on Ubuntu 16.04)
If you have a CUDA-capable GPU from NVIDIA and you want to use it you first have to follow the instructions on https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html to install CUDA.

pip install the following packages in a **python3** virtual environment:
* numpy
* opencv-python
* Keras (make sure it's 2.2.4 or higher)
* tensorflow (or tensorflow-gpu if you want to use CUDA)
* scikit-learn
* pandas
* matplotlib
* scikit-plot
* tensorflowjs (if you want to export models for the web)

Depending on what you want to do, the project expects the following folders to be present with their corresponding content:

### ../data
The data folder can be found on https://www.kaggle.com/rasmuspeterakerlund/balls-and-hands-in-videos-of-juggling and contains the dataset for training new localization models.

### ../grid_models
This folder can be found on https://www.kaggle.com/rasmuspeterakerlund/balls-and-hands-in-videos-of-juggling and contains pretrained models for localizing balls and hands.

### ../submovavg150
If you want to train your own models that predict on frames where the moving average of the previous frames has been subtracted from the current frame you have to create the folder ../submovavg150 manually and then run createsubmovavg.py

### ../patterns
The patterns folder can be found on https://www.kaggle.com/rasmuspeterakerlund/thirtysix-juggling-patterns and contains 36 common juggling patterns that you can use to train your own models for pattern categorization.

### ../pattern_models
This folder can be found on https://www.kaggle.com/rasmuspeterakerlund/thirtysix-juggling-patterns and contains pretrained pattern categorization models that are used by patterndetectdemo.py

## Troubleshooting
### Possible causes for low framerates
* The webcamera that is connected automatically lowers framerate to compensate for low lighting. Solution: Introduce more light or adjust webcam settings with tools such as guvcview.
* You use the flipping option on the models and use an intel-CPU without using their version of tensorflow. Solution: Turn flipping of or install the intel version of tensorflow: https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide
* You use a MAC and OpenCV has still not fixed the performance of cv2.waitKey() and/or cv2.imshow().
