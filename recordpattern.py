import numpy as np
import cv2
import time
import sys
import csv
from utils import handleTensorflowSession
from drawingutils import drawBallsAndHands
from gridmodel import GridModel
from frameratechecker import FramerateChecker

if(len(sys.argv) != 3):
    print("The first argument should be the filename for the recorded pattern.")
    print("The second argument should be number of balls to record.")
    print("Example: python " + sys.argv[0] + " mythreeballpattern.csv 3")
    sys.exit()

handleTensorflowSession(memoryLimit=0.2)

FRAMES = 30*100
FILENAME = sys.argv[1]
BALLS = int(sys.argv[2])
recording = np.zeros((FRAMES, 4+BALLS*2), dtype=np.uint8)
cap = cv2.VideoCapture(0)
framerateChecker = FramerateChecker(expected_fps=30)
gridModel = GridModel("../grid_models/grid_model_submovavg_64x64.h5", nBalls=BALLS, flip=False, postprocess=True)

print("Recording will be saved to: " + FILENAME)
for i in range(-150,FRAMES):
    framerateChecker.check()
    ret, img = cap.read()
    if not ret:
        print ("Couldn't get frame from camera.")
        break
    else:
        ballsAndHands = gridModel.predict(img)

        canvas = np.zeros((256,256,3))
        drawBallsAndHands(canvas, ballsAndHands)
        if i >= 0:
            recording[i,:2] = ballsAndHands["rhand"]
            recording[i,2:4] = ballsAndHands["lhand"]
            recording[i,4:] = ballsAndHands["balls"].flatten()

        cv2.imshow('RecordPattern', canvas)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

with open(FILENAME, mode='w') as pattern_file:
    pattern_writer = csv.writer(pattern_file,)
    for i in range(0,FRAMES):
        pattern_writer.writerow(recording[i])
