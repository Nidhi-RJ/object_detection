#live webcam detection for YOLO model
#the classes can be found in the mdeta_yolo file in the /computer-vision/datasets/global_daily_object_yolo/Dollar street trial path

import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train-4/weights/best.pt")

#shows all the classes the model has been trained on
print("Model classes: ", model.names)

CONFIDENCE_THRESHOLD = 0.60

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break 

    #we dont need to preprocess the image since YOLO preprocesses the image internally
    result = model.predict(
        source = frame,
        conf = CONFIDENCE_THRESHOLD
    )

    #draw predicted boxes, class names and confidence scores
    annotated_frame = result[0].plot()

    cv2.imshow("Object Detection", annotated_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()