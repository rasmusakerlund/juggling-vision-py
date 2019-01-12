import numpy as np
import cv2
import time
from utils import handleTensorflowSession
from drawingutils import drawBallsAndHands
from keras.models import load_model
from gridmodel import GridModel
from frameratechecker import FramerateChecker

handleTensorflowSession(memoryLimit=0.2)

gridModel = GridModel("../grid_models/grid_model_submovavg_64x64.h5")
patternModel = load_model("../pattern_models/3b_pattern_model.h5")
cap = cv2.VideoCapture(0)
history = []
framerateChecker = FramerateChecker(expected_fps=30)
names = ["441", "box", "cascade", "42, left hand", "shower, left hand", "mill's mess", "one up two up", "42, right hand", "reverse cascade", "shower, right hand", "takeouts", "tennis"]
while(True):
    framerateChecker.check()
    ret, original_img = cap.read()
    if not ret:
        print ("Couldn't get frame from camera.")
        break
    else:
        height, width, channels = original_img.shape
        tocrop = int((width - height) / 2)
        original_img = original_img[:,tocrop:-tocrop]
        ballsAndHands = gridModel.predict(original_img.copy())
        coordinates = []
        coordinates.extend(ballsAndHands["rhand"])
        coordinates.extend(ballsAndHands["lhand"])
        coordinates.extend(ballsAndHands["balls"].flatten())
        history.append(coordinates)
        if len(history) > 30:
            del history[0]
        else:
            continue

        pattern = np.array(history)
        pattern[:,::2] = pattern[:,::2] - np.mean(pattern[:,::2])
        pattern[:,1::2] = pattern[:,1::2] - np.mean(pattern[:,1::2])
        pattern = pattern / pattern.std()
        pattern = np.expand_dims(pattern, axis=0)
        pattern_activations = patternModel.predict(pattern)[0]

        img = np.zeros((256,512+128,3), dtype=np.uint8)
        for i in range(12):
            img[int(255-pattern_activations[i]*200):,256+i*32:256+i*32+32,:] = 100

        font = cv2.FONT_HERSHEY_SIMPLEX
        img[:,:256,:] = cv2.resize(original_img, (256,256), cv2.INTER_CUBIC)
        cv2.putText(img,names[np.argmax(pattern_activations)],(265,30), font, 1,(255,255,255),1,cv2.LINE_AA)
        drawBallsAndHands(img, ballsAndHands)
        img = cv2.resize(img, (768*2+256,768), cv2.INTER_CUBIC)


        cv2.imshow('Webcam', img)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
