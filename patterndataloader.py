from keras.utils import Sequence
from collections import OrderedDict
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
import cv2
import csv
import pandas as pd
import numpy as np
import random

class PatternDataLoader:
    def __init__(self, filename, length=60):
        self.length = length
        self.filename = filename
        self.patternsFolder = "../patterns/"
        self.trainx, self.trainy = self._getSet(0,2400)
        self.trainx = self._shuffleBalls(self.trainx)
        self.trainx, self.trainy = shuffle(self.trainx, self.trainy)
        self.valx, self.valy = self._getSet(2400,2700)
        self.testx, self.testy = self._getSet(2700,3000)

    def _getSet(self, start, stop):
        X = []
        y = []
        with open(self.patternsFolder + self.filename) as patternlist:
            count = 0
            for filename in patternlist:
                filename = filename.rstrip('\n')
                annotations = pd.read_csv(self.patternsFolder + filename, header=None).values
                annotations = annotations[start:stop:1]
                patterns = self._annotationsToX(annotations)
                X.extend(patterns)
                y.extend(np.full(patterns.shape[0], count))
                if filename[2] == 'L':
                    filename = filename[:2] + 'R' + filename[3:]
                elif filename[2] == 'R':
                    filename = filename[:2] + 'L' + filename[3:]
                annotations = pd.read_csv(self.patternsFolder + filename, header=None).values
                annotations = annotations[start:stop:1]
                annotations = self._flipAnnotations(annotations)
                patterns = self._annotationsToX(annotations)
                X.extend(patterns)
                y.extend(np.full(patterns.shape[0], count))
                count += 1
        return np.array(X), to_categorical(np.array(y))

    def _flipAnnotations(self, annotations):
        annotations[:,::2] = -annotations[:,::2]
        annotations[:,[0,1,2,3]] = annotations[:,[2,3,0,1]]
        return annotations

    def _annotationsToX(self, annotations):
        X = []
        for i in range(annotations.shape[0]-self.length+1):
            pattern = np.array(annotations[i:i+self.length], dtype=np.float32)
            pattern[:,::2] = pattern[:,::2] - np.mean(pattern[:,::2])
            pattern[:,1::2] = pattern[:,1::2] - np.mean(pattern[:,1::2])
            pattern = pattern / pattern.std()
            X.append(pattern)

        return np.array(X)

    def _shuffleBalls(self, clean_set):
        reshaped_set = np.reshape(clean_set, (len(clean_set), self.length, -1, 2))
        for i in range(len(reshaped_set)):
            for j in range(self.length):
                np.random.shuffle(reshaped_set[i,j,2:])
        clean_set = np.reshape(reshaped_set, (len(clean_set), self.length, -1))
        return clean_set

    def getNames(self):
        with open(self.patternsFolder + self.filename) as patternlist:
            names = []
            for filename in patternlist:
                filename = filename.rstrip('\n')
                names.append(filename[2:-4])
        return names
