import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Buffer for sequence (e.g., 30 frames)
sequence_buffer = deque(maxlen=30)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    
    # Draw landmarks
    if results.right_hand_landmarks or results.left_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    
    # Extract landmarks and add to buffer
    landmarks = []
    if results.right_hand_landmarks:
        landmarks.extend([(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark])
    else:
        landmarks.extend([(0, 0, 0)] * 21)  # Pad with zeros if no hand detected
    if results.left_hand_landmarks:
        landmarks.extend([(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark])
    else:
        landmarks.extend([(0, 0, 0)] * 21)
    
    sequence_buffer.append(landmarks)
    
    # Process sequence when buffer is full
    if len(sequence_buffer) == 30:
        sequence = np.array(sequence_buffer)
        # TODO: Feed 'sequence' into your trained model for prediction
        # Example: prediction = model.predict(sequence)
        print("Sequence ready for prediction:", sequence.shape)  # (30, 42, 3)
    
    # Show frame
    cv2.imshow("Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()