import csv
import cv2
import numpy as np
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import random

class JugglingDataLoader(Sequence):
    def __init__(self, shape=(64,64), timesteps=1, batch_size=8, gridShape=(15,15), expressFactor=1, imageGenerator=ImageDataGenerator(), dataType='BGR', nballs=[1,2,3]):
        self.batch_size = batch_size
        self.expressFactor = expressFactor
        self.dataType = dataType
        self.nballs = nballs
        self.channels = 3
        self.timesteps = timesteps
        self.gridShape = gridShape
        self.shape = shape
        self.dataFolder = "../data/"
        self.submovavg150 = "../submovavg150/"
        self.annotationsFolder = self.dataFolder + "annotations/"
        self.framesFolder = self.dataFolder + "frames/"
        self.allRows = []
        self.trainRows = self._loadSetRows("trainvideos", self.expressFactor)
        self.validationRows = self._loadSetRows("validationvideos", self.expressFactor)
        self.testRows = self._loadSetRows("testvideos")
        self.timestepShuffle()
        self.imageGenerator = imageGenerator

    def __len__(self):
        return len(self.trainRows) // self.batch_size

    def __getitem__(self, idx):
        images = np.zeros((self.batch_size,self.shape[0],self.shape[1],self.channels))
        grids = np.zeros((self.batch_size,*self.gridShape,9))
        for i in range(self.batch_size):
            localDict = self.imageGenerator.get_random_transform((256,256))
            row = self.trainRows[idx*self.batch_size+i]
            images[i] = self.getImage(row[0], transDict=localDict)
            grids[i] = self.getGrid(row, transDict=localDict)
        return images, grids

    def on_epoch_end(self):
        self.timestepShuffle()

    def _loadSetRows(self, filename, expressFactor=1):
        setRows = []
        with open(self.dataFolder + filename) as f:
            for videoline in f:
                videoline = videoline.rstrip('\n')
                if int(videoline[0]) not in self.nballs:
                    continue
                with open(self.annotationsFolder + videoline) as csvfile:
                    readCSV = csv.reader(csvfile, delimiter=',')
                    count = 0
                    for row in readCSV:
                        if count < 300 * expressFactor:
                            if not(self.dataType == "SUBMOVAVG" and count < 1):
                                setRows.append(row)
                                self.allRows.append(row)
                        count += 1
        return setRows

    def timestepShuffle(self, timesteps=False):
        if timesteps == False:
            timesteps = self.timesteps
        tmpList = [self.trainRows[i:i+timesteps] for i in range(0, len(self.trainRows), timesteps)]
        shuffle(tmpList)
        self.trainRows = []
        for lists in tmpList:
            for row in lists:
                self.trainRows.append(row)

    def streamTrainSet(self):
        for row in self.trainRows:
            yield cv2.imread(self.framesFolder + row[0]), self.getCoordinates(row)

    def streamValidationSet(self):
        for row in self.validationRows:
            yield cv2.imread(self.framesFolder + row[0]), self.getCoordinates(row)

    def streamTestSet(self):
        for row in self.testRows:
            yield cv2.imread(self.framesFolder + row[0]), self.getCoordinates(row)

    def streamAll(self):
        for row in self.allRows:
            yield cv2.imread(self.framesFolder + row[0]), self.getCoordinates(row)

    def getImage(self, filename, transDict=None):
        if self.dataType == 'BGR':
            return self.getBGR(filename, transDict)
        elif self.dataType == 'SUBMOVAVG':
            return self.getSubMovAvg(filename, transDict)
        assert False, 'Invalid choice of dataType.'

    def getBGR(self, filename, transDict=None):
        img = cv2.imread(self.framesFolder + filename)
        if transDict != None:
            img = self.transformImage(img, transDict)
        img = cv2.resize(img, self.shape, cv2.INTER_CUBIC)
        img = img - np.min(img)
        img = img / np.max(img)
        return img

    def getSubMovAvg(self, filename, transDict=None):
        img = cv2.imread(self.submovavg150 + filename)
        if transDict != None:
            img = self.transformImage(img, transDict)
        img = img.astype(np.float32) / 256
        img = cv2.resize(img, self.shape, cv2.INTER_CUBIC)
        return img

    def transformImage(self, img, transDict):
        return self.imageGenerator.apply_transform(img, transform_parameters=transDict)

    def getGrid(self, row, transDict=False):
        coordinates = self.getCoordinates(row, transDict)
        coordinates = coordinates / 256.
        gridWidth = self.gridShape[0]
        gridHeight = self.gridShape[1]
        boxWidth = 1. / gridWidth
        boxHeight = 1. / gridHeight
        grid = np.zeros((gridWidth, gridHeight, 9))
        for i in range(4, len(coordinates), 2):
            xIndex = int(coordinates[i] // boxWidth)
            yIndex = int(coordinates[i+1] // boxHeight)
            if xIndex >= 0 and xIndex < gridWidth and yIndex >= 0 and yIndex < gridHeight:
                grid[xIndex,yIndex,0] = 1
                grid[xIndex,yIndex,1] = (coordinates[i] - xIndex*boxWidth) / boxWidth
                grid[xIndex,yIndex,2] = (coordinates[i+1] - yIndex*boxHeight) / boxHeight
        for i in range(0, 4, 2):
            xIndex = int(coordinates[i] // boxWidth)
            yIndex = int(coordinates[i+1] // boxHeight)
            if xIndex >= 0 and xIndex < gridWidth and yIndex >= 0 and yIndex < gridHeight:
                grid[xIndex,yIndex,3+3*i//2] = 1
                grid[xIndex,yIndex,4+3*i//2] = (coordinates[i] - xIndex*boxWidth) / boxWidth
                grid[xIndex,yIndex,5+3*i//2] = (coordinates[i+1] - yIndex*boxHeight) / boxHeight
        return grid

    def getCoordinates(self, row, transDict=False):
        coordinates = []
        nballs = int(row[0][0])
        for b in range(nballs*2+4):
            coordinates.append(int(row[1+b]))
        coordinates = np.array(coordinates)
        if transDict != False:
            coordinates = self.transformCoordinates(coordinates, transDict)
        return coordinates

    def transformCoordinates(self, coordinates, transDict):
        for c in range(len(coordinates) // 2):
            coordinates[c*2+1] -= transDict["tx"]
            coordinates[c*2] -= transDict["ty"]
            coordinates[c*2+1] = (coordinates[c*2+1] -128 ) / transDict["zx"] + 128
            coordinates[c*2] = (coordinates[c*2] -128) / transDict["zy"] + 128
            if transDict["flip_horizontal"] == True:
                coordinates[c*2] = 255 - coordinates[c*2]
        if transDict["flip_horizontal"] == True:
            tmpHand = coordinates[0:2]
            coordinates[0:2] = coordinates[2:4]
            coordinates[2:4] = tmpHand
        return coordinates

    def getValidationSet(self):
        count = len(self.validationRows)
        images = np.zeros((count*2,self.shape[0],self.shape[1],self.channels))
        grids = np.zeros((count*2,*self.gridShape,9))
        i = 0
        for row in self.validationRows:
            images[i] = self.getImage(row[0])
            grids[i] = self.getGrid(row)
            i = i + 1
        transDict = {}
        transDict["flip_horizontal"] = True
        transDict["tx"] = 0
        transDict["ty"] = 0
        transDict["zx"] = 1
        transDict["zy"] = 1
        for row in self.validationRows:
            images[i] = self.getImage(row[0], transDict)
            grids[i] = self.getGrid(row, transDict)
            i = i + 1
        return images, grids
