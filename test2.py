import cv2
import numpy as np
from ultralytics import YOLO
model = YOLO(r"C:\Users\SOHAM MONDAL\Desktop\Yolov10Firedetectionmodel\runs\content\runs\detect\train\weights\best.pt")

cap = cv2.VideoCapture(0)

prev_fire_positions = []  
static_frames = 0  
STATIC_THRESHOLD = 20  # Number of frames fire must remain still to be considered fake

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.6)

    fire_detected = False
    new_fire_positions = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = box.conf[0].item()
            
            
            new_fire_positions.append((x1, y1, x2, y2))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Fire {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            fire_detected = True

    if fire_detected:
        if prev_fire_positions and new_fire_positions:
            
            if all(any(abs(x1 - px1) < 10 and abs(y1 - py1) < 10 for px1, py1, px2, py2 in prev_fire_positions) for x1, y1, x2, y2 in new_fire_positions):
                static_frames += 1
            else:
                static_frames = 0 
        prev_fire_positions = new_fire_positions 

        # If fire has been static for too long, mark it as fake
        if static_frames >= STATIC_THRESHOLD:
            cv2.putText(frame, "FAKE FIRE (Screen)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "REAL FIRE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Fire Detection - Static vs. Real", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
