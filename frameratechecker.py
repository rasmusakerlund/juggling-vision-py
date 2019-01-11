import cv2
import time

class FramerateChecker:
    def __init__(self, expected_fps=30, tolerance=0.1):
        self.expected_fps = expected_fps
        self.tolerance = tolerance
        self.oldTime = None
        self.i = 0

    def check(self):
        if self.i == 0:
            if self.oldTime == None:
                self.oldTime = time.time()
            else:
                newTime = time.time()
                diffTime = newTime - self.oldTime
                self.oldTime = newTime
                if diffTime > 1+self.tolerance:
                    print("Framerate low. " + str(self.expected_fps) + " frames took " + str(diffTime) + " seconds. If this message continues to be printed you should consult the README.")
                if diffTime < 1-self.tolerance:
                    print("Framerate high. " + str(self.expected_fps) + " frames took " + str(diffTime) + " seconds. If this message continues to be printed you should consult the README.")
        self.i = (self.i + 1) % self.expected_fps
