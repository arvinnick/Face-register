import cv2
import sqlite3
import numpy as np
import dlib
from datetime import datetime
import requests
import tkinter as tk
from tkinter import ttk
import threading
from tkinter import *
import datetime as dt
import uuid
import hashlib

def get_unique_id():
    global unique_id
    # Get the MAC address and format it
    mac = uuid.getnode()
    mac_formatted = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0, 2*6, 2)][::-1])
    unique_id = mac_formatted
    print(unique_id)
    return unique_id

get_unique_id()

def sent_AP():
    # Define the API endpoint
    url = "https://goodarzicc.ir/User/Arrival/Api"
    NationalCode = '0013407732'
    ArriveType = 'True'
    MacAddress = unique_id

    payload = {'NationalCode': NationalCode, 'ArriveType': ArriveType, 'MacAddress': MacAddress}

    # Define the headers (if needed)
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    try:
        # Send the POST request
        response = requests.post(url, data=payload, headers=headers)
        
        # Check the response status code
        if response.status_code == 200:
            print('Data sent successfully!')
            print('Response:', response.text)
            message_var.config(text=f"{response.text}")
        elif response.status_code == 400:
            print('Failed to send data.')
            print('Status code:', response.status_code)
            print('Response:', response.text)
            message_var.config(text=f"{response.text}")
    except requests.exceptions.RequestException as e:
        # Handle any other exceptions that may occur
        print('An error occurred:', str(e))
        message_var.config(text=f"An error occurred: {str(e)}")
        progress.stop()

# Initialize the database for recording presence with timestamps
conn_datetime = sqlite3.connect('Face_time_place.db', check_same_thread=False)
action = conn_datetime.cursor()
action.execute('''CREATE TABLE IF NOT EXISTS presence_time 
                  (id INTEGER PRIMARY KEY, datetime TIMESTAMP, person_name TEXT)''')

# Initialize the database for face encodings
conn = sqlite3.connect('face_data.db', check_same_thread=False)
cursor = conn.cursor()

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Set a threshold for the face recognition distance
RECOGNITION_THRESHOLD = 0.6

def is_blinking(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = cv2.norm(np.array(left_point) - np.array(right_point))
    ver_line_length = cv2.norm(np.array(center_top) - np.array(center_bottom))

    ratio = hor_line_length / ver_line_length
    return ratio

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def recognize_person():
    global person_name
    print("Starting recognition. Please look at the camera.")
    video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    known_face_encodings = []
    known_face_names = []

    cursor.execute('SELECT * FROM faces')
    rows = cursor.fetchall()

    for row in rows:
        name = row[1]
        encoding = np.frombuffer(row[2], dtype=np.float64)
        known_face_names.append(name)
        known_face_encodings.append(encoding)

    while True:
        ret, frame = video_capture.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 1)

        for face in faces:
            shape = sp(frame, face)
            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            face_encoding = np.array(face_descriptor)
            min_distance = float('inf')
            person_name = "Unknown"

            for known_face_encoding, known_face_name in zip(known_face_encodings, known_face_names):
                distance = np.linalg.norm(face_encoding - known_face_encoding)

                # Print debugging information
                print(f"Comparing to {known_face_name}: Distance = {distance}")

                if distance < min_distance:
                    min_distance = distance
                    if distance < RECOGNITION_THRESHOLD:
                        person_name = known_face_name

            left_eye_ratio = is_blinking([36, 37, 38, 39, 40, 41], shape)
            right_eye_ratio = is_blinking([42, 43, 44, 45, 46, 47], shape)
            blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if blink_ratio > 5.7:  # Adjust this threshold based on your testing
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, f"{person_name} ({min_distance:.2f})", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"Recognized: {person_name} with distance {min_distance}")
                current_time = datetime.now()
                action.execute('INSERT INTO presence_time (person_name, datetime) VALUES (?, ?)', (person_name, current_time))
                conn_datetime.commit()
                sent_AP()

                # Stop the recognition and progress bar if a face is recognized
                if person_name != "Unknown":
                    video_capture.release()
                    cv2.destroyAllWindows()
                    return

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Stopping recognition.")
    video_capture.release()
    cv2.destroyAllWindows()

def get_input():
    # Define the date to compare as a string
    date_string = '2025-06-30'

    # Convert the string to a datetime object
    date_to_compare = dt.datetime.strptime(date_string, '%Y-%m-%d')
    if dt.datetime.today() <= date_to_compare:
        progress.start()
        recognize_person()

def start_task():
    # Start the long-running task in a separate thread
    threading.Thread(target=get_input).start()

# Create the main window
root = tk.Tk()
root.title("Identity process")

# Create text widget and specify size.
T = Text(root, height = 5, width = 52)

# Create label
l = Label(root, text = "MacAddress")
l.config(font =("Courier", 14))

Fact = unique_id

l.pack()
T.pack()

# Insert The Fact.
T.insert(tk.END, Fact)

# Showing the progress 
progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress.pack(pady=20)

# Create a button that calls the get_input function when clicked
button = tk.Button(root, text="تشخيص چهره", command=start_task)
button.pack(pady=10)

# Create an output widget
message_var = tk.Message(root, text="", width=300)
message_var.pack(pady=20)

# Run the main event loop
root.mainloop()
