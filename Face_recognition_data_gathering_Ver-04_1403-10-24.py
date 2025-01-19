import cv2
import sqlite3
import os
import dlib
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
from tkinter import *
import datetime as dt

# Function to get input from the entry widget and save it as a variable
def get_input():
    date_string = '2025-06-30'
    date_to_compare = dt.datetime.strptime(date_string, '%Y-%m-%d')
    if dt.datetime.today() <= date_to_compare:
        global password
        password = password.get()
        if password == '1234':
            progress.start()
            conn = sqlite3.connect('face_data.db')
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY, national_code TEXT, name TEXT, encoding BLOB)''')
            conn.commit()

            detector = dlib.get_frontal_face_detector()
            sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

            def gather_data(name, national_code):
                print(f"Starting to gather data for {name}. Please look at the camera.")
                if not os.path.exists(f'dataset/{national_code}'):
                    os.makedirs(f'dataset/{national_code}')
                else:
                    gather_data_face = "The National code is already in the database. Please delete the file face_data.db if you want to replace data."
                    message_var.config(text=gather_data_face)
                    progress.stop()
                    return

                video_capture = cv2.VideoCapture(0)
                if not video_capture.isOpened():
                    print("Cannot open camera")
                    message_var.config(text="Cannot open camera")
                    progress.stop()
                    return

                count = 0

                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting ...")
                        message_var.config(text="Can't receive frame (stream end?). Exiting ...")
                        progress.stop()
                        break

                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray_frame, 1)

                    for face in faces:
                        count += 1
                        shape = sp(frame, face)
                        face_descriptor = facerec.compute_face_descriptor(frame, shape)
                        face_encoding = np.array(face_descriptor)
                        cursor.execute('INSERT INTO faces (national_code, name, encoding) VALUES (?, ?, ?)', (national_code, name, face_encoding.tobytes()))
                        conn.commit()
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                        cv2.putText(frame, f"Gathering data for {name} ({count}/20)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        print(f"Captured face {count} for {name}")

                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
                        break

                End = f"Finished gathering data for {name}."
                message_var.config(text=End)
                progress.stop()
                video_capture.release()
                cv2.destroyAllWindows()

            name = entry_name.get()
            national_code = entry_melicode.get()
            gather_data(name, national_code)
        else:
            pass_incor = "The password was incorrect! Please close the app and try again later."
            message_var.config(text=pass_incor)
            progress.stop()
    else:
        Licence = "No license for this date."
        message_var.config(text=Licence)
        progress.stop()

def start_task():
    threading.Thread(target=get_input).start()

root = tk.Tk()
root.title("Input specifications")

name_label = tk.Label(root, text='Enter the name of the person:', font=('calibre', 10, 'bold'))
name_label.pack(pady=10)

entry_name = tk.Entry(root, width=30)
entry_name.pack(pady=10)

name_label = tk.Label(root, text='Enter the national code of the person:', font=('calibre', 10, 'bold'))
name_label.pack(pady=10)

entry_melicode = tk.Entry(root, width=30)
entry_melicode.pack(pady=10)

name_label = tk.Label(root, text='Enter the password:', font=('calibre', 10, 'bold'))
name_label.pack(pady=10)

password = tk.Entry(root, width=30, show='*')
password.pack(pady=10)

progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress.pack(pady=20)

button = tk.Button(root, text="Submit", command=start_task)
button.pack(pady=10)

message_var = tk.Message(root, text="", width=300)
message_var.pack(pady=20)

root.mainloop()
