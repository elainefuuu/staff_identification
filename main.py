import cv2
from ultralytics import YOLO

path = "utils/sample.mp4"
model = YOLO('weights/yolo11l.onnx')    # exported model
classes = [0]     # staff

cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
print("streaming video")
frame_index = 0
skip_rate = 4

while cap.isOpened():
            success, frame = cap.read()
            # frame = cv2.resize(frame,(1280,720))

            if success:
                        result = model.track(frame, conf=0.1, classes=[0])
                        # print(result)
                        # print(result[0].boxes)
                        
                        if result[0].boxes.id is not None:
                            for idx, objectID in enumerate(result[0].boxes.id.tolist()):
                                x1, y1, x2, y2 = int(result[0].boxes.xyxy[idx][0]), int(result[0].boxes.xyxy[idx][1]), int(result[0].boxes.xyxy[idx][2]), int(result[0].boxes.xyxy[idx][3])

                                box = result[0].boxes.xyxy[idx].tolist()

                                cv2.putText(frame, f'Class:{result[0].names[int(result[0].boxes.cls[idx])]}', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)  
                                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

                                if result[0].boxes.cls.tolist()[idx] == 0:
                                       
                                    print(f"frame no.: {frame_index}, staff coor: {x1, y1, x2, y2}")                                   

                        cv2.imshow('Video Stream', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
                        frame_index += 1

            else:
                    print("Error: Could not read frame")
                    while True:
                        cap = cv2.VideoCapture(path)
                        if cap.isOpened() == True:
                            print(cap.isOpened())
                            break

cap.release()
cv2.destroyAllWindows()