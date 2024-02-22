import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QCheckBox, QHBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
import imutils
from imutils.video import VideoStream

import detect_face

class VideoCaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.video_stream = VideoStream(src=0, framerate=10).start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Facial Detection is Running")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        self.button = QPushButton("Start Detection", self)
        self.button.clicked.connect(self.start_detection)
        self.layout.addWidget(self.button)

        self.checkboxes_layout = QHBoxLayout()
        self.check_eyes = QCheckBox("Eyes Detection", self)
        self.check_nose = QCheckBox("Nose Detection", self)
        self.check_mouth = QCheckBox("Mouth Detection", self)
        self.check_eyes.setChecked(True)
        self.check_nose.setChecked(True)
        self.check_mouth.setChecked(True)
        self.checkboxes_layout.addWidget(self.check_eyes)
        self.checkboxes_layout.addWidget(self.check_nose)
        self.checkboxes_layout.addWidget(self.check_mouth)
        self.layout.addLayout(self.checkboxes_layout)

        self.central_widget.setLayout(self.layout)

    def update_frame(self):
        frame = self.video_stream.read()
        frame = imutils.resize(frame, width=800)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

    def start_detection(self):
        frame = self.video_stream.read()
        frame2 = detect_face.detect_image(frame, self.check_eyes.isChecked(), self.check_nose.isChecked(),
                                                   self.check_mouth.isChecked())

        try:
            if frame2[0] == None:
                cv2.imshow("Faces detected", frame)
        except:
            inx=1
            for frame_in in frame2:
                cv2.imshow(f"Faces detected {inx}", frame_in)
                inx+=1
            print(f"{inx-1} faces has been detected")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoCaptureApp()
    window.show()
    sys.exit(app.exec_())
