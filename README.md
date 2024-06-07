# Hand Tracking Project

This project uses MediaPipe and OpenCV for real-time hand tracking and labeling via a webcam.

## Installation

```bash
pip install mediapipe opencv-python
```
## Usage
1. Import Libraries:
```python
import mediapipe as mp
import cv2
import numpy as np
```
2. Initialize MediaPipe:
```python
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
```
3. Define Hand Label Function:
```python
def get_label(index, hand, handedness):
    label = handedness.classification[0].label
    score = handedness.classification[0].score
    text = '{} {}'.format(label, round(score, 2))
    coords = tuple(np.multiply(
        np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
        [640, 480]).astype(int))
    return text, coords
```
4. Start Video Capture and Hand Tracking:
```python
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
                text, coord = get_label(idx, hand, handedness)
                cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()
```
You can stop the video by clicking ```q``` button
