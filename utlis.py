import numpy as np
import cv2, datetime
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt


# Image Processing

# 1 PreProcess img
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray
    imgBlur = cv2.GaussianBlur(imgGray, (13, 13), 0)  # blur 5, 5, 1
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # threshold
    return imgThreshold

# 2 Biggest contour
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)  # perimeter
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # number of points
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

# 3. Reoder points
def reoder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

# 4. Split Boxes
def splitBoxes(input_img):
    rows = np.vsplit(input_img, 3)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 3)
        for box in cols:
            boxes.append(box)
    return boxes

# 5. Convert cv2 img to QtPixMap
def cvtQtimg(img, w=384, h=216):
    """get cv2 img and return pixmap for Qt"""
    frame = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = QImage(frame, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img)



# Game and UI

# 1. Status
def show_msg(app=None, msg="", stop=True):
    if stop: app.Stop_Game()
    app.lbl_status.setText(msg)
    print(msg)

# 2. About
class dlg_about(QDialog):   
    def __init__(self, parent=None):
        super(dlg_about, self).__init__(parent)
        # Display the about window
        self = uic.loadUi('ui/dlg_about.ui', self)
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle("About")
        self.setWindowIcon(QIcon("./ui/icon.png"))
        self.dummy_3.setText(''.join(chr(ord(char) - 1) for char in "Efwfmpqfe!cz;!Nbmblb!E/Hvobxbsebob/"))
        pixmap = QPixmap("./ui/icon.png")
        pixmap = pixmap.scaledToWidth(200, Qt.SmoothTransformation)
        self.icon.setPixmap(pixmap)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_ok.setDefault(True)
        self.show()


#..................................................................................................................
def training(boxes):
    for j in range(9):
        pre_fix = 'train/' + str(datetime.datetime.now()).replace(":", "_").replace("-", "_").replace(" ", "_")
        cv2.imwrite(pre_fix + '_' + str(j) + '.jpg', boxes[j])
        print(pre_fix + '_' + str(j) + '.jpg')
#..................................................................................................................