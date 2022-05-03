# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:57:29 2022

@author: Ivan
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 900)
        MainWindow.setMinimumSize(QtCore.QSize(700, 500))
        MainWindow.setMaximumSize(QtCore.QSize(1600, 900))
        MainWindow.setStyleSheet("QWidget#centralwidget{background-color: rgb(85, 170, 127);}\n" "")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(300, 70, 1280, 720 ))
        #self.label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label.setText("")
        self.label.setObjectName("label")
        #self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        #self.pushButton.setGeometry(QtCore.QRect(400, 800, 231, 41))
        #self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        #self.pushButton.setStyleSheet("border-radius:20px;\n" "background-color: rgb(232, 58, 58);")
        #self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(400,800, 231, 41))
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.setStyleSheet("border-radius:20px;\n" "background-color: rgb(85, 255, 127);\n" "\n" "")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(980, 800, 231, 41))
        self.pushButton_3.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_3.setStyleSheet("border-radius:20px;\n" "background-color: rgb(85, 170, 255);")
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(170, 20, 341, 31))
        self.label_2.setStyleSheet("font: 14pt \"MS Shell Dlg 2\";")
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(200, 70, 12, 72))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        #self.pushButton.clicked.connect(self.cancel)
        self.pushButton_2.clicked.connect(self.start_video)
        self.pushButton_3.clicked.connect(self.on_click)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def start_video(self):
        self.Work = Work()
        self.Work.start()
        self.Work.Imageupd.connect(self.Imageupd_slot)
        
         

    def Imageupd_slot(self, Image):
        self.label.setPixmap(QPixmap.fromImage(Image))

    def cancel(self):
        self.label.clear()
        self.Work.stop()

    '''def salir(self):
        sys.exit()'''
    
    #@pyqtSlot()
    def on_click(self):
        print('PyQt5 button click')

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        #self.pushButton.setText(_translate("MainWindow", "Stop"))
        self.pushButton_2.setText(_translate("MainWindow", "Start"))
        self.pushButton_3.setText(_translate("MainWindow", "Seleccionar y visualizar video"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Modelo</p></body></html>"))

class Work(QThread):
    
    Imageupd = pyqtSignal(QImage)
    
    def run(self):
        emotion_model_path = './models/emotion_model.hdf5'
        emotion_labels = get_labels('fer2013')
        frame_window = 10
        emotion_offsets = (20, 40)
        face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
        emotion_classifier = load_model(emotion_model_path)
        emotion_target_size = emotion_classifier.input_shape[1:3]
        emotion_window = []
        
        
        self.hilo_corriendo = True
        cap = cv2.VideoCapture(0)
        
        while self.hilo_corriendo:
            ret, frame = cap.read()
            if ret:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
            			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

                for face_coordinates in faces:
                    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                    gray_face = gray_image[y1:y2, x1:x2]
                    try:
                        gray_face = cv2.resize(gray_face, (emotion_target_size))
                    except:
                        continue
                    
                    gray_face = preprocess_input(gray_face, True)
                    gray_face = np.expand_dims(gray_face, 0)
                    gray_face = np.expand_dims(gray_face, -1)
                    emotion_prediction = emotion_classifier.predict(gray_face)
                    emotion_probability = np.max(emotion_prediction)
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_text = emotion_labels[emotion_label_arg]
                    emotion_window.append(emotion_text)

                    if len(emotion_window) > frame_window:
                        emotion_window.pop(0)
                    try:
                        emotion_mode = mode(emotion_window)
                    except:
                        continue
                    
                    if emotion_text == 'angry':
                        color = emotion_probability * np.asarray((255, 0, 0))
                    elif emotion_text == 'sad':
                        color = emotion_probability * np.asarray((0, 0, 255))
                    elif emotion_text == 'happy':
                        color = emotion_probability * np.asarray((255, 255, 0))
                    elif emotion_text == 'surprise':
                        color = emotion_probability * np.asarray((0, 255, 255))
                    else:
                        color = emotion_probability * np.asarray((0, 255, 0))

                    color = color.astype(int)
                    color = color.tolist()

                    draw_bounding_box(face_coordinates, Image, color)
                    draw_text(face_coordinates, Image, emotion_mode, color, 0, -45, 1, 1)
                    
                frame = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
                
                
                
                flip = cv2.flip(Image, 1)
                convertir_QT = QImage(flip.data, flip.shape[1], flip.shape[0], QImage.Format_RGB888)
                pic = convertir_QT.scaled(1280, 720, Qt.KeepAspectRatio)
                self.Imageupd.emit(pic)
                
                #cap.release()
                cv2.destroyAllWindows()
    
    '''def stop(self):
        self.hilo_corriendo = False
        self.quit()'''
                

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())