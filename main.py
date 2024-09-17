import cv2
import os
import supervision as sv
from ultralytics import YOLO

model = YOLO(f'best.pt')
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VedioCapture(0)

if not cap.isOpened():
    print("Unable to read camera feed")

img_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    annoatated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annoatated_image = label_annotator.annotate(scene=annoatated_image, detections=detections)
    cv2.imshow('Webcam', annoatated_image)

    k = cv2.waitKey(1)

    if k%256 == 27:
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()


