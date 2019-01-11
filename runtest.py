import numpy as np
import cv2
from gridmodel import GridModel
from utils import handleTensorflowSession
from testutils import ScoreKeeper
from drawingutils import drawBallsAndHands
from jugglingdataloader import JugglingDataLoader


handleTensorflowSession(memoryLimit=0.2)

def runTest(model):
    model.summary()
    print("\tball_acc\thand_acc\tframerate")
    dataLoader = JugglingDataLoader()
    scoreKeeper = ScoreKeeper()

    count = 0
    for frame, true_coordinates in dataLoader.streamTestSet():
        if count % 300 == 0:
            model.reset(nBalls=len(true_coordinates)//2-2)
        count += 1

        ballsAndHands = model.predict(frame)
        ballsAndHands = scoreKeeper.score(ballsAndHands, true_coordinates)
        drawBallsAndHands(frame, ballsAndHands)

        frame = cv2.resize(frame, (512, 512))
        cv2.imshow('Testing model', frame)
        cv2.waitKey(1)

    scoreKeeper.printAverage()
    print()

runTest(GridModel('../grid_models/grid_model_submovavg_64x64_light.h5', preprocessType="SUBMOVAVG", flip=False, postprocess=True))
#runTest(GridModel('../grid_models/grid_model_submovavg_64x64.h5', preprocessType="SUBMOVAVG", flip=False, postprocess=True))
#runTest(GridModel('../grid_models/grid_model_submovavg_128x128.h5', preprocessType="SUBMOVAVG", flip=False, postprocess=True))
