import cv2
import numpy as np
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from losses import grid_loss_with_hands
from preprocessing import MovingAveragePreprocessor
from postprocessing import BallsAndHandsPostprocessor, gridToBallsAndHands, flipGrid


class GridModel:
    def __init__(self, filename, nBalls=3, preprocessType="SUBMOVAVG", flip=False, postprocess=True):
        self.filename = filename
        self.preprocessType=preprocessType
        self.flip = flip
        self.postprocess = postprocess
        self.model = load_model(filename, custom_objects={'grid_loss_with_hands': grid_loss_with_hands})
        self.input_shape = self.model.layers[0].input_shape[1:3]
        self.reset(nBalls)

    def summary(self):
        flipStr = "FLIP" if self.flip else "NOFLIP"
        postStr = "POSTPROCESS" if self.postprocess else "NOPOSTPROCESS"
        print(self.filename, self.preprocessType, flipStr, postStr, 'CPU')

    def reset(self, nBalls):
        self.nBalls = nBalls
        if self.preprocessType == "SUBMOVAVG":
            self.preprocessor = MovingAveragePreprocessor(0.150)
        if self.postprocess == True:
            self.postprocessor = BallsAndHandsPostprocessor(self.nBalls)

    def predict(self, frame):
        height, width, channels = frame.shape
        if height != width:
            tocrop = int((width - height) / 2)
            frame = frame[:,tocrop:-tocrop]
        frame = cv2.resize(frame, self.input_shape)

        if self.preprocessType == "SUBMOVAVG":
            frame = self.preprocessor.process(frame)
        else:
            frame = frame - np.min(frame)
            frame = frame / (np.max(frame) + 0.00001)

        if self.flip:
            tmp = np.zeros((2,self.input_shape[0],self.input_shape[1],3))
            tmp[0] = frame
            tmp[1] = cv2.flip(frame, 1)
            grids = self.model.predict(tmp)
            grid = (grids[0] + flipGrid(grids[1])) / 2
        else:
            grid = self.model.predict(np.expand_dims(frame, axis=0))[0]

        ballsAndHands = gridToBallsAndHands(grid, self.nBalls)
        if self.postprocess:
            ballsAndHands = self.postprocessor.process(ballsAndHands)

        return ballsAndHands
