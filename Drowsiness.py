import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys

# Use winsound for Windows, pygame for other OS
if sys.platform.startswith("win"):
    import winsound
    def beep():
        winsound.Beep(1500, 500)  # Frequency = 1500 Hz, Duration = 500ms
else:
    import pygame
    pygame.mixer.init()
    alarm_sound = "alarm.wav"  # Use a valid alarm sound file
    if not os.path.exists(alarm_sound):
        print("Error: Alarm sound file not found!")
    def beep():
        pygame.mixer.music.load(alarm_sound)
        pygame.mixer.music.play()

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eye landmarks (indices from Mediapipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_landmarks):
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))  # Vertical line 1
    B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))  # Vertical line 2
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))  # Horizontal line

    ear = (A + B) / (2.0 * C)  # EAR formula
    return ear

cap = cv2.VideoCapture(0)
drowsy_time = 0
alarm_triggered = False  # Prevent continuous alarm

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status = "Alert"
    color = (0, 255, 0)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract eye landmarks
            left_eye = [(int(face_landmarks.landmark[i].x * frame.shape[1]), int(face_landmarks.landmark[i].y * frame.shape[0])) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * frame.shape[1]), int(face_landmarks.landmark[i].y * frame.shape[0])) for i in RIGHT_EYE]

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Draw eye landmarks
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Drowsiness Detection
            if avg_ear < 0.25:  # Eyes closed
                if drowsy_time == 0:
                    drowsy_time = time.time()
                elif time.time() - drowsy_time > 2:  # If closed for 2+ sec
                    status = "DROWSY! WAKE UP!"
                    color = (0, 0, 255)

                    if not alarm_triggered:  # Play alarm only once
                        beep()
                        alarm_triggered = True
            else:
                drowsy_time = 0
                alarm_triggered = False  # Reset alarm trigger

    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Drowsiness Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
