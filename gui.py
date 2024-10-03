import tkinter
import cv2
from tkinter import *
import mediapipe as mp
from PIL.Image import fromarray
from PIL import Image, ImageTk
import numpy as np
from keras import models



#loading model
model=models.load_model("my_model_2.h5")


#32 alphanet we have, used to plug the output from model to get the alphabet
alphabet = [
    'ع', 'الـ', 'أ', 'ب', 'د', 'ظ', 'ض', 'ف', 'ق',
    'غ', 'هـ', 'ح', 'ج', 'ك', 'خ', 'لا', 'ل',
    'م', 'ن', 'ر', 'ص', 'س', 'ش', 'ط', 'ت',
    'ث', 'ذ', 'ة', 'و', 'ئ', 'ي', 'ز'
]
#calling mediapipe to draw the lines
mp_drawing=mp.solutions.drawing_utils
mp_Hands=mp.solutions.hands
hand=mp_Hands.Hands()

#function used to use the model that we got and send frames to it to determine the alphabet
def detection_sings(frame):
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    resize=cv2.resize(grey,(64,64))
    normlize=resize/255
    reshape=np.reshape(normlize,(1,64,64,1))
    result=model.predict(reshape)
    label= np.argmax(result,axis=1)[0]
    return alphabet[label]


#gui build
root = Tk()
root.title('Sign Language')
root.geometry('1000x1000')
frame =tkinter.Frame(root)
label1 =tkinter.Label(frame,width=900 , height=600)
label2=tkinter.Label( frame)

frame.pack()
label1.pack()
label2.pack()


#reading camera
cap =cv2.VideoCapture(0)

#function to read camera input and send frames to detection_sings function and printing the output on the screen
def update_frame():
    ret, img = cap.read()
    if ret:
        output = detection_sings(img)
        rgb=cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
        result=hand.process(rgb)
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img,hand_landmark,mp_Hands.HAND_CONNECTIONS)
        img= cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img= fromarray(img)
        imgtk=ImageTk.PhotoImage(image=img)
        label1.imgtk=imgtk
        label1.configure( image= imgtk)
        label2.configure(text=output,pady=100,font=("Helvetica", 30), fg="red")
        cv2.destroyAllWindows()
    root.after(10,update_frame)


#calling thr gui and the method to start working
update_frame()
root.mainloop()
