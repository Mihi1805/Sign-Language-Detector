import cv2

print("Attempting to force camera connection...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam. Check permissions.")
else:
    print("Webcam connected successfully! Press 'q' on the window to close.")
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Force Permission Window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()