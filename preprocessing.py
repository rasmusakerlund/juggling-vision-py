import numpy as np

class MovingAveragePreprocessor:
    def __init__(self, updateFactor=0.150):
        self.updateFactor = updateFactor
        self.avgImg = None

    def process(self, img):
        img = img - np.min(img)
        img = img / (np.max(img) + 0.00001)
        if self.avgImg is None:
            self.avgImg = img
        self.avgImg = (1-self.updateFactor)*self.avgImg + self.updateFactor*img
        self.avgImg = self.avgImg - np.min(self.avgImg)
        self.avgImg = self.avgImg / (np.max(self.avgImg) + 0.00001)
        returnFrame = img - self.avgImg
        returnFrame = returnFrame - np.min(returnFrame)
        returnFrame = returnFrame / (np.max(returnFrame) + 0.00001)
        return returnFrame

def normalizeFrame(frame):
    frame = frame.astype(np.float32)
    frame = frame - np.min(frame)
    frame = frame / np.max(frame)
    return frame
