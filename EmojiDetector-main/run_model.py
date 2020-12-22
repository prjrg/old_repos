import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np

from define_model import define_network

model = define_network(48)
model.load_weights('data/model.h5')

emotion_dict = {0: "    Angry    ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
emoji_dict = {0:"./emojis/angry.png",2:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surprised.png"}

global last_frame
last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
global cap
show_text=[0]

def show_vid(lmain):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("cant open the camera")
    flag, frame = cap.read()
    frame = cv2.resize(frame, (600, 500))
    bounding_box = cv2.CascadeClassifier('/home/pedro/Documents/EmojiDetector/venv/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y - 50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y: y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)

        maxindex = int(np.argmax(prediction))

        cv2.putText(frame, emoji_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex

    if flag is None:
        print("Major error")
    elif flag:
        global last_frame
        last_frame = frame.copy()
        pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_vid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

def show_vid2(lmain2, lmain3):
    frame1 = cv2.imread(emoji_dict[show_text[0]])
    pic1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    img1 = Image.fromarray(frame1)
    imgtk1 = ImageTk.PhotoImage(image=img1)
    lmain2.imgtk1 = imgtk1
    lmain3.configure(text=emotion_dict[show_text[0]], font=('arial',45,'bold'))
    lmain2.configure(image=imgtk1)
    lmain2.after(10, show_vid2)


if __name__ == '__main__':
    root = tk.Tk()
    img = ImageTk.PhotoImage(Image.open("logo.png"))
    heading = Label(root, image=img, bg='black')

    heading.pack()
    heading2 = Label(root, text="Photo to Emoji", pady=20, font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')

    heading2.pack()
    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)

    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'
    exitbutton = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)
    show_vid(lmain)
    show_vid2(lmain2, lmain3)
    root.mainloop()