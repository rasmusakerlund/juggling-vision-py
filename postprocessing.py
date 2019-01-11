import numpy as np
from utils import getDistance
import itertools



class SingleObjectStreakPostprocessor:
    def __init__(self):
        self.previousPosition = None
        self.streak = 0
        self.streakPosition = None
        self.timeSinceStreak = 0
        self.maxDistance = getDistance(0,0,256,256) * 0.1

    def getPreviousPosition(self):
        return self.previousPosition

    def process(self, position):
        if self.previousPosition is not None:
            if getDistance(position[0], position[1], self.previousPosition[0], self.previousPosition[1]) < self.maxDistance:
                self.streak += 1
                if self.streak >= 1:
                    self.streakPosition = position
                    self.timeSinceStreak = 0
            else:
                self.streak = 0
                self.timeSinceStreak += 1

        self.previousPosition = position
        if self.streakPosition is not None and self.timeSinceStreak < 2:
            return self.streakPosition
        else:
            return position

class BallMatcher:
    def __init__(self):
        self.previousBalls = None

    def process(self, balls):
        if self.previousBalls is None:
            self.previousBalls = balls
            return balls
        else:
            bestDistance = 100000000
            bestPermutation = None
            for permutation in itertools.permutations(balls):
                distance = 0
                for i in range(len(permutation)):
                    distance += getDistance(permutation[i][0], permutation[i][1], self.previousBalls[i][0], self.previousBalls[i][1])
                if distance < bestDistance:
                    bestPermutation = permutation
                    bestDistance = distance
            self.previousBalls = bestPermutation
            return np.array(self.previousBalls)


class BallsAndHandsPostprocessor:
    def __init__(self, nballs):
        self.matcher = BallMatcher()
        self.rhand = SingleObjectStreakPostprocessor()
        self.lhand = SingleObjectStreakPostprocessor()
        self.balls = []
        for i in range(nballs):
            self.balls.append(SingleObjectStreakPostprocessor())
        self.nballs = nballs

    def process(self, bnh):
        bnh["rhand"] = self.rhand.process(bnh["rhand"])
        bnh["lhand"] = self.lhand.process(bnh["lhand"])
        bnh["balls"] = self.matcher.process(bnh["balls"])

        for i in range(self.nballs):
            bnh["balls"][i] = self.balls[i].process(bnh["balls"][i])
        return bnh

def flipGrid(grid):
    grid[:,:,1::3] = 1 - grid[:,:,1::3]
    grid = np.flip(grid, 0)
    grid[:,:,[3,4,5,6,7,8]] = grid[:,:,[6,7,8,3,4,5]]
    return grid

def gridToBallsAndHands(grid, nballs):
    bnh = {}
    bnh["balls"] = np.reshape(gridToBalls(grid, nballs=nballs), (-1,2))
    bnh["rhand"], bnh["lhand"] = gridToHands(grid)
    bnh["checked"] = False
    return bnh

def gridToBalls(grid, nballs, otherHand=None):
    gridSurface = grid[:,:,0]
    gridWidth = grid.shape[0]
    gridHeight = grid.shape[1]
    minDistance = getDistance(0,0,256,256)*0.04
    boxWidth = 256 // gridWidth
    boxHeight = 256 // gridHeight
    ballList = []
    for i in range(nballs):
        while True: # While ball candidate not accepted
            x, y = np.unravel_index(np.argmax(gridSurface), (gridWidth,gridHeight))
            ballX = int(boxWidth*x + boxWidth*grid[x,y,1])
            ballY = int(boxHeight*y + boxHeight*grid[x,y,2])
            gridSurface[x,y] = -1
            acceptable = True
            for j in range(0,len(ballList),2):
                if getDistance(ballX,ballY,ballList[j],ballList[j+1]) < minDistance:
                    acceptable = False
            if otherHand is not None:
                if getDistance(ballX,ballY,otherHand[0], otherHand[1]) < minDistance:
                    acceptable = False
            if acceptable:
                ballList.append(ballX)
                ballList.append(ballY)
                break
    return np.array(ballList)

def gridToHands(grid):
    if np.max(grid[...,3]) > np.max(grid[...,6]):
        rhand = gridToBalls(grid[...,3:6], 1)
        lhand = gridToBalls(grid[...,6:9], 1, rhand)
    else:
        lhand = gridToBalls(grid[...,6:9], 1)
        rhand = gridToBalls(grid[...,3:6], 1, lhand)
    return rhand, lhand
