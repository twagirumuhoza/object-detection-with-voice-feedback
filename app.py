import numpy as np
import cv2
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Detection parameters
thres = 0.5          # Minimum detection confidence
nms_threshold = 0.2  # Non-maximum suppression threshold

# Setup video capture
cap = cv2.VideoCapture(0)  # 0 for built-in webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)

# Load class names from file (one per line)
classNames = []
with open('objects.txt', 'r') as f:
    classNames = f.read().splitlines()
print("Class names:", classNames)

# Drawing parameters
font = cv2.FONT_HERSHEY_PLAIN
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

# Load the pre-trained DNN model
weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Dictionary to track last announced time for each object
last_announced = {}  # Format: {object_name: last_announcement_time}
announcement_interval = 30  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection on the current frame
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)

    # Process detections if available
    if classIds is not None and len(classIds) > 0:
        # Convert confidences to a 1D list and process bounding boxes
        confs = np.array(confs).reshape(1, -1)[0]
        confs = list(map(float, confs))
        bbox = list(bbox)
        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
        if indices is None:
            indices = []
        current_time = time.time()

        for i in indices:
            # Handle index whether it's a scalar or an array
            idx = int(i) if np.isscalar(i) else int(i[0])
            box = bbox[idx]
            x, y, w, h = box[0], box[1], box[2], box[3]
            confidence = round(confs[idx], 2)
            object_name = classNames[classIds[idx] - 1]
            color = Colors[classIds[idx] - 1]

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{object_name} ", (x + 10, y + 20), font, 1, color, 2)

            # Check if the object is new or if it hasn't been announced in the last 30 sec.
            if object_name not in last_announced or (current_time - last_announced[object_name] >= announcement_interval):
                message = f"Detected object is  {object_name} "
                print(message)  # Output to console
                engine.say(message)
                engine.runAndWait()
                last_announced[object_name] = current_time

    # Show the video feed
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
