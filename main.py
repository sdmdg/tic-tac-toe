import sys, cv2, threading
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread
from tf_scan import tf_scan
from utlis import *
from board import *
import mediapipe as mp

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('ui/main.ui', self)
        self.setWindowIcon(QIcon("./ui/icon.png"))

        # User Inputs
        self.fps = int(self.input_fps.text())
        self.input_fps.textChanged.connect(lambda : self.ui_input_change())
        self.anti_cheat = False
        self.chk_box_anti_cheat.stateChanged.connect(lambda : self.ui_input_change())
        self.is_autoscan = False
        self.chk_box_auto_scan.stateChanged.connect(lambda : self.ui_input_change())
        self.waitTime = 1000    # in miliseconds

        self.is_playing = False
        self.frame = None
        self.frame_warp = None
        self.previous_board_imgs = []
        self.is_hand_detected = None
        
        self.counter = 0
        self.waitTime = int((self.waitTime/1000)*self.fps)

        self.load_res()
        self.load_tf_model()
        self.load_mediapipe()
        self.update_board()

        self.btn_toggle_preview.clicked.connect(lambda: self.start_camera_live_feed(id=int(self.input_cam_id.text())))
        self.btn_calibrate.clicked.connect(self.camera_live_feed_calibrateFrame)

        self.btn_game_start.clicked.connect(self.Start_Game)
        self.btn_game_scan.clicked.connect(lambda: self.Scan(is_forced=True))
        self.btn_game_reset.clicked.connect(self.Reset_Game)
        self.btn_about.clicked.connect(self.show_about)

        self.show()


    # UI
    def ui_input_change(self):
        self.anti_cheat = self.chk_box_anti_cheat.isChecked()
        self.is_autoscan = self.chk_box_auto_scan.isChecked()
        if self.anti_cheat:
            show_msg(app=gui, msg="If you're using this feature, make sure to add ai's moves to the physical board somehow.\nOtherwise this feature won't work properly.", stop=False)
        try:
            self.fps = int(self.input_fps.text())
        except:
            self.fps = 15

    def load_res(self):
        self.img_1 = QPixmap("ui/1.png").scaledToWidth(100, Qt.SmoothTransformation)
        self.img_0 = QPixmap("ui/0.png").scaledToWidth(100, Qt.SmoothTransformation)
        self.img_N = QPixmap("ui/N.png").scaledToWidth(100, Qt.SmoothTransformation)

        self.btn_calibrate.setEnabled(False)
        self.btn_game_start.setEnabled(False)
        self.btn_game_scan.setEnabled(False)
        self.btn_game_reset.setEnabled(False)
        self.lbl_hands.setVisible(False)

    def update_board(self):
        for id in range(9):
            match board.get_state(id):
                case 1: eval("self.lbl_grid_"+str(id)+".setPixmap(self.img_1)")
                case 0: eval("self.lbl_grid_"+str(id)+".setPixmap(self.img_0)")
                case None: eval("self.lbl_grid_"+str(id)+".setPixmap(self.img_N)")

    def show_about(self):
        dialog = dlg_about()
        _ = dialog.exec_()


    # Game
    def Start_Game(self):
        self.Reset_Game()
        self.is_playing = True
        self.chk_box_anti_cheat.setEnabled(False)
        self.Scan(pre_scan=True, is_forced=True)

    def Reset_Game(self):
        self.is_playing = False
        self.btn_game_start.setEnabled(True)
        self.btn_game_scan.setEnabled(False)
        self.btn_game_reset.setEnabled(False)
        self.chk_box_anti_cheat.setEnabled(True)
        board.reset()
        self.update_board()
        self.ui_input_change()
    
    def Stop_Game(self):
        self.is_playing = False
        self.is_autoscan = False
        self.btn_game_start.setEnabled(False)
        self.btn_game_scan.setEnabled(False)
    
    def Scan(self, pre_scan=False, is_forced=False):

        self.boxes = splitBoxes(self.frame_warp)
        #training(self.boxes)   # For training the keras model
        if is_forced:
            ids=[i for i in range(9)]
        else:
            ids=[]
            for id, image in enumerate(self.boxes):
                try:
                    if not self.find_differences(self.previous_board_imgs[id], image):
                        ids.append(id)
                except:
                    ids=[i for i in range(9)]

        self.previous_board_imgs = self.boxes
        _scanner = threading.Thread(target=self.Scanner.scan, args=[self.frame_warp, board, ids, self, pre_scan])
        _scanner.start()


    # Camera live feed
    def start_camera_live_feed(self, id=0):
        self.capture = cv2.VideoCapture(id) #"Comp 1.mp4"
        self.thread_camera_live = worker_camera_live_feed()
        self.thread_camera_live.start()
        self.btn_toggle_preview.setEnabled(False)
        self.input_cam_id.setEnabled(False)
        self.btn_calibrate.setEnabled(True)      

    def camera_live_feed_update(self):
        ret, self.frame = self.capture.read()
        self.lbl_video_live.setPixmap(cvtQtimg(self.frame))
        try:
            self.frame_warp = preview_board(self.frame)
            self.lbl_board.setPixmap(cvtQtimg(self.frame_warp, w=181, h=181))
        except:pass
        self.thread_hand_detection = worker_hand_detection(frame=self.frame)
        self.thread_hand_detection.start()
        if self.is_hand_detected:
            self.counter = 0
        else:
            if self.counter == self.waitTime:
                if self.is_autoscan and self.is_playing:
                    print("Auto Scanning...")
                    self.Scan()
            self.counter += 1


    # Image processing
    def load_tf_model(self):
        print("Loading Model")
        self.Scanner = tf_scan(model="model.h5", labels="labels.txt")

    def load_mediapipe(self):
        print("Loading Model")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

    def camera_live_feed_calibrateFrame(self):
        try: 
            calibrate_board(self.frame)
            self.btn_game_start.setEnabled(True)
        except Exception as e: print("Error-calibrateframe:", e)

    def find_differences(self, image1, image2):
        # Convert images to grayscale for pixel-wise comparison
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Threshold the difference image
        _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Check if there are any differences
        return cv2.countNonZero(thresholded) == 0


    # Output
    def send_output(self, ai_move):
        print(f"Ai move {ai_move}")
        #   Put your code here

    def on_hands_detected(self):
        #   Put your code here
        pass



class worker_camera_live_feed(QThread):
    def __init__(self):
        super().__init__()

    def run(self):
        while True:
            gui.camera_live_feed_update()
            cv2.waitKey(int(1/gui.fps * 1000))

class worker_hand_detection(QThread):
    def __init__(self, frame):
        super().__init__()
        self.frame = frame

    def run(self):
        rgb_frame = cv2.cvtColor(gui.frame, cv2.COLOR_BGR2RGB)
        results = gui.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            gui.is_hand_detected = True
            gui.lbl_hands.setVisible(True)
            gui.on_hands_detected()
        else:
            gui.is_hand_detected = False
            gui.lbl_hands.setVisible(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    board = Board()
    gui = MainWindow()
    sys.exit(app.exec_())