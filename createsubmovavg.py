import csv
import cv2
import numpy as np
from preprocessing import MovingAveragePreprocessor

bgrFolder = "../data/frames/"
submovavgFolder = "../submovavg150/"

count = 0
TOTAL_FRAMES = 54000
for setname in ["trainvideos", "validationvideos", "testvideos"]:
    with open("../data/" + setname) as f:
        for videoline in f:
            videoline = videoline.rstrip('\n')
            with open("../data/annotations/" + videoline) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                filenames = []
                preprocessor = MovingAveragePreprocessor(0.150)
                for row in readCSV:
                    filename = row[0]
                    frame = cv2.imread(bgrFolder + filename)
                    frame = preprocessor.process(frame)
                    frame = (frame*255).astype(np.uint8)
                    ret_imwrite = cv2.imwrite(submovavgFolder + filename, frame)
                    assert ret_imwrite, "Couldn't write frame. Have you created the directory " + submovavgFolder + "?"
                    count += 1
                    print("%d/%d\r" % (count, TOTAL_FRAMES), end='')
print()
