import numpy as np
import cv2
import itertools
import math
import time
from utils import getDistance

class ScoreKeeper:
    def __init__(self, ignore=30):
        self.ignore = ignore
        self.ignored = 0
        self.framesPerVideo = 300 - self.ignore
        self.totalFrames = 0
        self.totalBalls = 0
        self.totalHands = 0
        self.accurateFrames = 0
        self.accurateBalls = 0
        self.accurateHands = 0
        self.totalVFrames = 0
        self.totalVBalls = 0
        self.totalVHands = 0
        self.accurateVFrames = 0
        self.accurateVBalls = 0
        self.accurateVHands = 0
        self.timePassed = 0

    def score(self, bnh, true_coordinates):
        bnh = checkBallsAndHands(bnh, true_coordinates)
        if self.ignored < self.ignore:
            self.ignored += 1
            if self.ignored == self.ignore:
                self.starttime = time.time()
            return bnh
        self.totalVFrames += 1
        self.totalVBalls += len(bnh["balls"])
        self.totalVHands += 2
        self.accurateVBalls += len(bnh["validBalls"])
        self.accurateVHands += len(bnh["validRight"]) + len(bnh["validLeft"])
        if len(bnh["invalidBalls"]) + len(bnh["invalidRight"]) + len(bnh["invalidLeft"]) == 0:
            self.accurateVFrames += 1

        if self.totalVFrames == self.framesPerVideo:
            stoptime = time.time()
            VtimePassed = stoptime - self.starttime
            self.timePassed += VtimePassed
            self.totalFrames += self.totalVFrames
            self.totalBalls += self.totalVBalls
            self.totalHands += self.totalVHands
            self.accurateFrames += self.accurateVFrames
            self.accurateBalls += self.accurateVBalls
            self.accurateHands += self.accurateVHands
            accuracy = "Video %02d\t%.1f\t%.1f\t%d" % (self.totalFrames//self.framesPerVideo, self.accurateVBalls/self.totalVBalls*100, self.accurateVHands/self.totalVHands*100, self.framesPerVideo/VtimePassed)
            print(accuracy)
            self.totalVFrames = 0
            self.totalVBalls = 0
            self.totalVHands = 0
            self.accurateVFrames = 0
            self.accurateVBalls = 0
            self.accurateVHands = 0
            self.ignored = 0
        return bnh

    def printAverage(self):
        accuracy = "Average\t%.1f\t%.1f\t%d" % (self.accurateBalls/self.totalBalls*100, self.accurateHands/self.totalHands*100, self.totalFrames/self.timePassed)
        print(accuracy)



def checkBallsAndHands(bnh, true_coordinates):
    bnh["validBalls"], bnh["invalidBalls"] = isValidDetections(target_coordinates=true_coordinates[4:], pred_coordinates=bnh["balls"])
    bnh["validRight"], bnh["invalidRight"] = isValidDetections(target_coordinates=true_coordinates[0:2], pred_coordinates=bnh["rhand"])
    bnh["validLeft"], bnh["invalidLeft"] = isValidDetections(target_coordinates=true_coordinates[2:4], pred_coordinates=bnh["lhand"])
    bnh["checked"] = True
    return bnh

def isValidDetections(target_coordinates, pred_coordinates):
    bestDistance = 100000000
    for permutation in itertools.permutations(pred_coordinates.reshape(-1,2)):
        distance = 0
        for i in range(len(permutation)):
            distance += getDistance(permutation[i][0], permutation[i][1], target_coordinates[i*2], target_coordinates[i*2+1])
        if distance < bestDistance:
            bestPermutation = permutation
            bestDistance = distance

    valid = []
    invalid = []
    limit = getDistance(0,0,256,256) * 0.05
    for i in range(len(bestPermutation)):
        distance = getDistance(bestPermutation[i][0], bestPermutation[i][1], target_coordinates[i*2], target_coordinates[i*2+1])
        if distance < limit:
            valid.append(bestPermutation[i][0])
            valid.append(bestPermutation[i][1])
        else:
            invalid.append(bestPermutation[i][0])
            invalid.append(bestPermutation[i][1])
    return np.reshape(np.array(valid), (-1,2)), np.reshape(np.array(invalid), (-1,2))
