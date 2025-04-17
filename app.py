import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from kivy.clock import Clock
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

KV = '''
ScreenManager:
    MenuScreen:
    GameScreen:

<MenuScreen>:
    name: "menu"
    BoxLayout:
        orientation: "vertical"
        MDLabel:
            text: "Welcome to the Sign Language Learning Game!"
            halign: "center"
        MDRaisedButton:
            text: "Start Game"
            pos_hint: {"center_x": 0.5, "center_y": 0.5}
            on_release: root.manager.current = "game"

<GameScreen>:
    name: "game"
    BoxLayout:
        orientation: "vertical"
        MDLabel:
            id: target_letter_label
            text: "Target Letter: None"
            halign: "center"

        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: None
            height: '60dp'
            MDRaisedButton:
                text: "A"
                on_release: app.set_target_letter('A')
            MDRaisedButton:
                text: "B"
                on_release: app.set_target_letter('B')
            MDRaisedButton:
                text: "C"
                on_release: app.set_target_letter('C')

        MDRaisedButton:
            text: "Start Capturing"
            pos_hint: {"center_x": 0.5}
            on_release: app.start_capturing()
        MDRaisedButton:
            text: "Return to Menu"
            pos_hint: {"center_x": 0.5}
            on_release: root.manager.current = "menu"
'''


class MenuScreen(Screen):
    pass


class GameScreen(Screen):
    pass


class SignLanguageApp(MDApp):
    def build(self, *args):
        self.detector = HandDetector(maxHands=2)
        self.classifier = Classifier("/Users/sunguangtian/Desktop/HandSignDectection/Model/keras_model.h5",
                                     "/Users/sunguangtian/Desktop/HandSignDectection/Model/labels.txt")
        self.targetLetter = None
        self.score = 0
        self.cap = cv2.VideoCapture(0)
        self.event = None

        return Builder.load_string(KV)

    def set_target_letter(self, letter):
        """Set the target letter based on the button pressed."""
        self.targetLetter = letter

    def start_capturing(self):
        if not self.event:
            self.event = Clock.schedule_interval(self.capture, 1.0 / 30.0)  # 30 FPS

    def capture(self, dt):
        success, img = self.cap.read()
        if not success:
            print("Failed to capture image. Check your camera.")
            return

        hands, img = self.detector.findHands(img)
        imgWhite = np.ones((300, 300, 3), np.uint8) * 255
        offset = 20

        if hands:
            min_x = min([hand["bbox"][0] for hand in hands])
            min_y = min([hand["bbox"][1] for hand in hands])
            max_x = max([hand["bbox"][0] + hand["bbox"][2] for hand in hands])
            max_y = max([hand["bbox"][1] + hand["bbox"][3] for hand in hands])

            w = max_x - min_x
            h = max_y - min_y
            aspectRatio = h / w

            if 0.5 <= aspectRatio <= 2:
                imgCrop = img[min_y - offset:max_y + offset, min_x - offset:max_x + offset]

                if aspectRatio > 1:
                    wCal = int(300 * w / h)
                    imgCrop = cv2.resize(imgCrop, (wCal, 300))
                    wGap = (300 - wCal) // 2
                    imgWhite[:, wGap:wCal + wGap] = imgCrop
                else:
                    hCal = int(300 * h / w)
                    imgCrop = cv2.resize(imgCrop, (300, hCal))
                    hGap = (300 - hCal) // 2
                    imgWhite[hGap:hCal + hGap, :] = imgCrop

                prediction, index = self.classifier.getPrediction(imgWhite)
                print(f"Predicted: {prediction}")  # Debugging statement

                if prediction == self.targetLetter:
                    print("Correct prediction!")  # Debugging statement
                    self.score += 10

        else:
            print("No hands detected.")  # Debugging statement

        # Update Kivy interface
        screen = self.root.get_screen("game")
        screen.ids.target_letter_label.text = f"Target Letter: {self.targetLetter} - Score: {self.score}"

        # Display target letter and score on the captured image
        cv2.putText(img, f"Target: {self.targetLetter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Score: {self.score}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the updated image in the window
        cv2.imshow("Sign Language Game", img)
        cv2.waitKey(1)

    def on_stop(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    SignLanguageApp().run()