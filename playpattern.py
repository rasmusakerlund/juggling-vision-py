import sys
import cv2
import pandas as pd
import numpy as np

print (len(sys.argv))
if len(sys.argv) != 2:
    print (sys.argv[0] + " expects one csv-file as its argument.")
    sys.exit()

recording = pd.read_csv(sys.argv[1], header=None).values

for i in range(0,recording.shape[0]):
    canvas = np.zeros((256,256,3), dtype=np.uint8)
    cv2.line(canvas, (recording[i,0]-10, recording[i,1]), (recording[i,0]+10, recording[i,1]), (0,255,0), 2)
    cv2.line(canvas, (recording[i,2]-10, recording[i,3]), (recording[i,2]+10, recording[i,3]), (0,0,255), 2)
    for j in range(4, recording.shape[1], 2):
        colorshift = j*50 % 255
        cv2.circle(canvas, (recording[i,j], recording[i,j+1]), 10, (colorshift,255-colorshift,colorshift), 2)
    cv2.imshow('PlayPattern', canvas)
    cv2.waitKey(15)
