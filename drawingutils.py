import cv2

def drawBallsAndHands(img, bnh):
    if bnh["checked"]:
        for b in range(bnh["validBalls"].shape[0]):
            cv2.circle(img, (bnh["validBalls"][b,0], bnh["validBalls"][b,1]), 10, (0,255,0), 2)
        for b in range(bnh["invalidBalls"].shape[0]):
            cv2.circle(img, (bnh["invalidBalls"][b,0], bnh["invalidBalls"][b,1]), 10, (0,0,255), 2)
        if len(bnh["validRight"]) != 0:
            cv2.line(img, (bnh["rhand"][0]-10, bnh["rhand"][1]), (bnh["rhand"][0]+10, bnh["rhand"][1]), (0,255,0), 2)
        else:
            cv2.line(img, (bnh["rhand"][0]-10, bnh["rhand"][1]), (bnh["rhand"][0]+10, bnh["rhand"][1]), (0,255,255), 2)
        if len(bnh["validLeft"]) != 0:
            cv2.line(img, (bnh["lhand"][0]-10, bnh["lhand"][1]), (bnh["lhand"][0]+10, bnh["lhand"][1]), (0,0,255), 2)
        else:
            cv2.line(img, (bnh["lhand"][0]-10, bnh["lhand"][1]), (bnh["lhand"][0]+10, bnh["lhand"][1]), (0,255,255), 2)
    else:
        for b in range(bnh["balls"].shape[0]):
            cv2.circle(img, (bnh["balls"][b,0], bnh["balls"][b,1]), 10, (0,255,0), 2)
        cv2.line(img, (bnh["rhand"][0]-10, bnh["rhand"][1]), (bnh["rhand"][0]+10, bnh["rhand"][1]), (0,255,0), 2)
        cv2.line(img, (bnh["lhand"][0]-10, bnh["lhand"][1]), (bnh["lhand"][0]+10, bnh["lhand"][1]), (0,0,255), 2)
