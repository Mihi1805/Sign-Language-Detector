import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model = tf.keras.models.load_model('sign_language_cnn.h5')
CATEGORIES = ['Sign A', 'Sign B']

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    
    # Match the (42, 2) zero structure used during data collection
    two_hands_landmarks = np.zeros((42, 2))
    
    if detection_result.hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks[:2]):
            for joint_idx, lm in enumerate(hand_landmarks):
                row_target = (hand_idx * 21) + joint_idx
                two_hands_landmarks[row_target] = [lm.x, lm.y]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (255, 0, 0), -1)
                
        # Format into a batch element: shape (1, 42, 2)
        input_data = np.array([two_hands_landmarks])
        
        prediction = model.predict(input_data, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class] * 100
        
        display_text = f"{CATEGORIES[predicted_class]} ({confidence:.1f}%)"
        cv2.putText(frame, display_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
    cv2.imshow('Two-Hand Live Classifier', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()