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
        if password == '4749':
            progress.start()
            conn = sqlite3.connect('face_data.db')
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY, national_code TEXT, encoding BLOB)''')
            conn.commit()

            try:
                detector = dlib.get_frontal_face_detector()
                sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
                facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
            except Exception as e:
                print(f"Error loading dlib models: {e}")
                return


##            detector = dlib.get_frontal_face_detector()
##            sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
##            facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

            def gather_data(national_code):
                print(f"Starting to gather data for {national_code}. Please look at the camera.")
                if not os.path.exists(f'dataset/{national_code}'):
                    os.makedirs(f'dataset/{national_code}')
                else:
                    gather_data_face = "اين کد ملي قبلا ثبت شده است"
                    message_var.config(text=gather_data_face)
                    progress.stop()
                    return

                video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                if not video_capture.isOpened():
                    print("Cannot open camera")
                    message_var.config(text="Cannot open camera")
                    progress.stop()
                    return
                
                else:
                    print("Camera initialized successfully.")

                count = 0

                while True:
                    ret, frame = video_capture.read()
                    if not ret or frame is None:
                        print("Can't receive frame (stream end?). Exiting ...")
                        message_var.config(text="Can't receive frame (stream end?). Exiting ...")
                        progress.stop()
                        break
                    
                    if frame is not None and frame.dtype != np.uint8:
                        print("Frame dtype is not uint8. Skipping this frame.")
                        continue

                    fps = video_capture.get(cv2.CAP_PROP_FPS)
                    print(f"FPS: {fps}")


                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Debugging log
                    print(f"Original frame type: {frame.dtype}, shape: {frame.shape}")
                    print(f"Gray frame type: {gray_frame.dtype}, shape: {gray_frame.shape}")
                    gray_frame = np.array(gray_frame, dtype=np.uint8)
                    faces = detector(gray_frame, 1)

                    for face in faces:
                        count += 1
                        shape = sp(frame, face)
                        face_descriptor = facerec.compute_face_descriptor(frame, shape)
                        face_encoding = np.array(face_descriptor)
                        cursor.execute('INSERT INTO faces (national_code, encoding) VALUES (?, ?)', (national_code, face_encoding.tobytes()))
                        conn.commit()
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                        cv2.putText(frame, f"Gathering data for {national_code} ({count}/20)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        print(f"Captured face {count} for {national_code}")

                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
                        break

                End = f"تصاوير مربوط به  {national_code} با موفقيت ذخيره گرديد"
                message_var.config(text=End)
                progress.stop()
                video_capture.release()
                cv2.destroyAllWindows()


            national_code = entry_melicode.get()
            gather_data(national_code)
        else:
            pass_incor = "رمز ورود اشتباه است لطفا از برنامه بسته و مجدد باز نماييد"
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


name_label = tk.Label(root, text=':لطفا کد ملي را وارد نماييد', font=('calibre', 10, 'bold'))
name_label.pack(pady=10)

entry_melicode = tk.Entry(root, width=30)
entry_melicode.pack(pady=10)

name_label = tk.Label(root, text=':لطفا رمز را وارد نماييد', font=('calibre', 10, 'bold'))
name_label.pack(pady=10)

password = tk.Entry(root, width=30, show='*')
password.pack(pady=10)

progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress.pack(pady=20)

button = tk.Button(root, text="ثبت چهره", command=start_task)
button.pack(pady=10)

message_var = tk.Message(root, text="", width=300)
message_var.pack(pady=20)

root.mainloop()
