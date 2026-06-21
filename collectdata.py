import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_DIR = './hand_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 1. Increase num_hands to 2
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
GESTURES = ['Sign_A', 'Sign_B']

for gesture in GESTURES:
    collected_data = []
    print(f"\n--- GET READY FOR TWO-HANDED: {gesture} ---")
    print("Press 'c' to start capturing 100 frames, or 'q' to quit.")
    
    capturing = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = detector.detect(mp_image)
        
        # Initialize an empty matrix for 2 hands (42 joints total, x and y)
        two_hands_landmarks = np.zeros((42, 2))
        
        if detection_result.hand_landmarks:
            # Loop through up to 2 detected hands
            for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks[:2]):
                for joint_idx, lm in enumerate(hand_landmarks):
                    # Hand 0 fills slots 0-20; Hand 1 fills slots 21-41
                    row_target = (hand_idx * 21) + joint_idx
                    two_hands_landmarks[row_target] = [lm.x, lm.y]
                    
                    # Draw dots on screen
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
            
            if capturing:
                collected_data.append(two_hands_landmarks)
                cv2.putText(frame, f"Capturing: {len(collected_data)}/100", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if len(collected_data) >= 100: break
        
        if not capturing:
            cv2.putText(frame, f"Press 'c' to record {gesture}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow('Two-Hand Data Collector', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'): capturing = True
        elif key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); exit()

    np.save(os.path.join(DATA_DIR, f'{gesture}.npy'), np.array(collected_data))
    print(f"Saved 2-hand array for {gesture}.")

cap.release()
cv2.destroyAllWindows()