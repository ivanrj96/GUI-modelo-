# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:04:13 2022

@author: Ivan
"""

from cgitb import text
from distutils import command
#from msilib.schema import ListBox
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import time
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

import tkinter
import customtkinter


# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
window_frame = 10
emotion_offsets = (20, 40)

# loading model
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# Proceso para el contador
proceso=0

# Guardado de datos
f = open("SaveEmotions.csv","w+")


def iniciar_camara():
    global cap
    cap = cv2.VideoCapture(0)
    visualizar()
    
def elegir_visualizar_video():
    global cap

    if cap is not None:
        lblVideo.image = ""
        cap.release()
        cap = None

    path_video = filedialog.askopenfilename(filetypes = [
        ("all video format", ".mp4"),
        ("all video format", ".avi"),
        ("all video format", ".mov"),
        ("all video format", ".wmv"),
        ("all video format", ".mkv"),
        ("all video format", ".flv"),
        ("all video format", ".f4v"),
        ("all video format", ".swf"),
        ("all video format", ".webm"),
        ("all video format", ".m4v"),
        ("all video format", ".html5"),
        ("all video format", ".avchd")])
    if len(path_video) > 0:
        btnVisualizar.configure(state=tkinter.DISABLED)
        lblInfoVideoPath.configure(text=path_video)
        cap = cv2.VideoCapture(path_video)
        pathInputVideo = "..." + path_video[-20:]
        lblInfoVideoPath.configure(text=pathInputVideo)
        cap = cv2.VideoCapture(path_video)
        visualizar()
    
def visualizar():
    global cap
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = deteccion_facilal(frame)
        frame = imutils.resize(frame, width=720, height=480)
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)
        iniciar()
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(10, visualizar)
    else:
        lblVideo.image = ""
        lblInfoVideoPath.configure(text="")
        
        cap.release()
        cv2.destroyAllWindows()

# Metodo para detectar la expresion
def deteccion_facilal(frame):
    # Counters
    AngerCounter = 0;
    SadCounter = 0;
    HappyCounter = 0;
    SurpriceCounter = 0;
    
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        if len(emotion_window) > window_frame:
            emotion_window.pop(0)
            
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
            AngerCounter += 1
            #print("Times Anger detected: ",  AngerCounter)
            f.write(" Numero de eventos Enojo: %d\r\n " % AngerCounter)
            listBox.insert(0," Numero de eventos Enojo: %d\r\n " % AngerCounter, proceso)
            
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
            SadCounter += 1
            #print("Times Sad detected", SadCounter)
            f.write(" Numero de eventos Triste: %d\r\n " % AngerCounter)
            listBox.insert(0," Numero de eventos Triste: %d\r\n " % AngerCounter, proceso)
            
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
            HappyCounter += 1
            #print("Times Happiness detected",HappyCounter)
            f.write(" Numero de eventos Felicidad: %d\r\n " % AngerCounter)
            listBox.insert(0," Numero de eventos Felicidad: %d\r\n " % AngerCounter, proceso)
            
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
            SurpriceCounter += 1
            #print("Times surprice detected", SurpriceCounter)
            f.write(" Numero de eventos Sorpresa: %d\r\n " % AngerCounter)
            listBox.insert(0," Numero de eventos Sorpresa: %d\r\n " % AngerCounter, proceso)
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)
        
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        frame = bgr_image
        
    return frame

# Método para iniciar el contador
def iniciar(h=0, m=0, s=0):
    global proceso
    #Verificamos si los segundos y los minutos son mayores a 60
    #Verificamos si las horas son mayores a 24
    if s >= 60:
        s=0
        m=m+1
        if m >= 60:
            m=0
            h=h+1
            if h >= 24:
                h=0
    #etiqueta que muestra el cronometro en pantalla
    time['text'] = str(h)+":"+str(m)+":"+str(s)
    # iniciamos la cuenta progresiva de los segundos
    proceso=time.after(2000, iniciar, (h), (m), (s+1))
            
def finalizar():
    
    parar()
    lblVideo.image = ""
    lblInfoVideoPath.configure(text="")
    btnIniciar.configure(state=tkinter.NORMAL)
    btnVisualizar.configure(state=tkinter.NORMAL)
    cap.release()
    
# Método para detener el contador
def parar():
    global proceso
    time.after_cancel(proceso)

cap = None
root = customtkinter.CTk()
root.title('Detector')

customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("green")  # Themes: blue (default), dark-blue, green

root.geometry("720x480")

lblInfo1 = customtkinter.CTkLabel(root, text="Facial Expression Detection", text_color="green", text_font=("italic","14"))
lblInfo1.grid(column=0, row=0, columnspan=2)
#lblInfo1.configure(bg='#008B8B')

btnIniciar = customtkinter.CTkButton(root, text="Iniciar camara", width=20, text_font=("italic"), text_color="black", fg_color='#008000', command=iniciar_camara)
btnIniciar.grid(column=0, row=1, pady= 10)

btnVisualizar = customtkinter.CTkButton(root, text="Elegir y visualizar video", width=20, text_font=("italic"), text_color="black", fg_color='#008000', command=elegir_visualizar_video)
btnVisualizar.grid(column=1, row=1, padx= 10, pady= 5, columnspan=4)

btnFinalizar = customtkinter.CTkButton(root, text="Finalizar", width=20, text_font=("italic"),fg_color='#008000',text_color="black", command=finalizar)
btnFinalizar.grid(column=0, row=4, columnspan=2, pady=10)



lblVideo = customtkinter.CTkLabel(root, text="")
lblVideo.grid(column=0, row=3, columnspan=2)

#lblInfo1 = customtkinter.CTkLabel(root, text="Video de entrada:")
#lblInfo1.grid(column=0, row=1)

lblInfoVideoPath = customtkinter.CTkLabel(root, text="Aún no se ha seleccionado un video", text_color="green", width=20)
lblInfoVideoPath.grid(column=0, row=2, columnspan=3, padx=40)

# Creación de ListBox para visualizar los eventos detectados
listBox = Listbox(root, font=("italic","12"))
listBox.insert(0,"")
listBox.place(width=250, height=565, x=850, y=84)
listBox.configure(bg='#008000', fg='#000000')

# Creación Label del contador
#text_var = tkinter.StringVar(value="")
time = customtkinter.CTkLabel(root, text_font=("18"), text="", text_color="green", width=20)
time.place(x=850, y=37)
#time.configure(bg='#008B8B')

root.mainloop()
