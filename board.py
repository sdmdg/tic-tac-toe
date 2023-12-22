from utlis import *
import numpy as np
import cv2

# Disable scientific notation
np.set_printoptions(suppress=True)
widthImg = heightImg = 450

def calibrate_board(img):
    global pts1
    # 1.Prepare the IMG
    imgThreshold = preProcess(img)

    # 2.Find all Contours
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3.Find the biggest Contour
    biggest, maxArea = biggestContour(contours)
    if biggest.size != 0:
        biggest = reoder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 24)
        pts1 = np.float32(biggest)

def preview_board(frame):
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(frame, matrix, (widthImg, heightImg))
    #cv2.imshow('Board', imgWarpColored)
    return imgWarpColored


# Data - class Board
class Board():
    def __init__(self):
        self.reset()

    def get_state(self, pos):
        return self.current_board_state[pos]
    
    def toString(self):
        return ",".join(str(i) for i in self.current_board_state)
    
    def is_ready(self):
        if ",".join(str(i) for i in self.current_detections) == ("None,"*9)[:-1]:
            return True
        else:
            return False

    def update(self, pos, val):
        self.previous_board_state[pos] = self.current_board_state[pos]
        self.current_board_state[pos] = val
        self.unavailable_pos.append(pos)

    def force_update(self, val):
        self.current_board_state = val
    
    def get_differences(self, board=[], output_ids=False):
        diffs = []
        for id, val in enumerate(board):
            if val != self.current_board_state[id]:
                diffs.append(id)
            board
        if output_ids:
            return diffs
        else:
            return len(diffs)
    
    def reset(self):
        self.previous_board_state = [None,None,None,None,None,None,None,None,None]
        self.current_board_state = [None,None,None,None,None,None,None,None,None]
        self.current_detections = [None,None,None,None,None,None,None,None,None]
        self.unavailable_pos = []