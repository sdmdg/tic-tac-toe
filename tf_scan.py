from keras.models import load_model
import numpy as np
import cv2
from PyQt5.QtCore import QThread
from utlis import splitBoxes, show_msg
from solver import find_ai_move

class tf_scan():
    def __init__(self, model, labels):
        # Load the model
        self.model = load_model(model, compile=False)
        self.class_names = ""

    def scan(self, frame, board, ids=[0,1,2,3,4,5,6,7,8], app=None, pre_scan=False):
        self.ids = ids
        self.app = app
        self.board = board
        self.is_pre_scan = pre_scan
        self.scanner = Scanner(frame=frame, ids=ids, model=self.model, class_names=self.class_names, board=board, app=app)
        self.scanner.start()
        self.scanner.wait()

        if self.is_pre_scan:
            self.pre_scan()
        else:
            self.validate_results()

    def pre_scan(self):
        print("Pre scan compleate.")
        show_msg(app=self.app, stop=False, msg="Go ahead :)")
        if self.board.is_ready():
            self.app.is_playing = True
            self.app.btn_game_start.setEnabled(False)
            self.app.btn_game_scan.setEnabled(True)
            self.app.btn_game_reset.setEnabled(True)
        else:
            show_msg(app=self.app, stop=False, msg="Please clean the board.")
            self.app.is_playing = False
            self.app.Reset_Game()
        self.app.update_board()

    def validate_results(self):
        absolute_changes = 0
        player_move = None
        if not self.app.anti_cheat:
            for id in self.board.unavailable_pos:
                self.ids.remove(id)

        for id in self.ids:
            old = self.board.current_board_state[id]
            new = self.board.current_detections[id]
            if old != new:
                absolute_changes += 1
                if (old == False and new != False):
                    print(f"You Cheated. ({id}) - from O to *")
                    show_msg(app=self.app, msg=f"You Cheated. ({id}) - from O to *")
                    return
                elif (old == True and new != True):
                    print(f"You Cheated. ({id}) - from X to *")
                    show_msg(app=self.app, msg=f"You Cheated. ({id}) - from X to *")
                    return
                else:
                    self.board.update(id, self.board.current_detections[id])
                    if new == False: 
                        player_move = id

        if absolute_changes > 1:
            print(f"You Cheated :(\n{absolute_changes} moves detected.")
            show_msg(app=self.app, msg=f"You Cheated :(\n{absolute_changes} moves detected.")
            return
        else:
            print(f"Player move is {player_move}")
            try:
                ai_move = find_ai_move(self.board.current_board_state, player_move, app=self.app)
                if ai_move != None:
                    self.board.update(ai_move, True)
                    self.app.send_output(ai_move)
            except:pass

        #print("Current board state : ", self.board.current_board_state)    # Debug
        self.app.update_board()

        if len(self.board.unavailable_pos) == 8:
            show_msg(app=self.app, msg="Well, it's a drow. Good job. :)")
            print("drow.")


class Scanner(QThread):
    def __init__(self, frame, ids, model, class_names, board, app):
        super().__init__()
        self.frame = frame
        self.ids = ids
        self.model = model
        self.class_names= class_names
        self.results = [None,None,None,None,None,None,None,None,None]
        self.board = board
        self.boxes = app.boxes

    def run(self):
        print("Scanner is running...")
        print("Scanning ids :", self.ids)
        if self.boxes == None:
            self.boxes = splitBoxes(self.frame)
        # Scanner
        for id, image in enumerate(self.boxes):
            # Skip unwanted boxes
            if id not in self.ids:
                continue
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            # Normalize the image array
            image = (image / 127.5) - 1
            # Predicts the model
            prediction = self.model.predict(image)
            index = np.argmax(prediction)
            confidence_score = prediction[0][index]

            if index == 0:              # None
                self.results[id] = None
            elif index == 1:            # Circle - False
                self.results[id] = False
            elif index == 2:            # Cross - True
                self.results[id] = True

        self.board.current_detections = self.results
        #print("Current detections : ", self.board.current_detections)  # Debug