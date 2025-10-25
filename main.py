import cv2
import cvzone
import math
from ultralytics import YOLO
from sort import *
import numpy as np
import os

video_path = 'people.mp4'
mask_path = 'mask_people.png'


#video setup
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error : Could not open the video file {video_path}")
    exit()

#Load YOLO model
model = YOLO('yolov8n.pt')
target_class_index = 0
        
CONFIDENCE_THRESHOLD = 0.4
#preaparing the mask
mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

success,first_frame = cap.read()
if not success:
    print("Error: Could not find the first frame")
    exit()
H,W,_ = first_frame.shape

if mask is not None:
    mask_final = cv2.resize(mask,(W,H))
    print("Mask loaded and resized")
else:
    mask_final=None
    print("Warning: Mask not found.Processing full frames.")

#initialize the tracker
tracker = Sort(max_age = 20,min_hits=3,iou_threshold=0.3)

#setup counting line & Counter
x1,y1 = 292,18
x2,y2= 292,1055
limits = [x1,y1,x2,y2]

totalCount=set()

#start the main loop

while True:
    success,img = cap.read()
    if not success:
        print("End of video orerror reading frame.")
        break
    #mask
    if mask_final is not None:
        imgRegion = cv2.bitwise_and(img,img,mask=mask_final)
        process_img = imgRegion
    else:
        imgRegion = img
        process_img = img

    #run yolo detections
    try:
        results = model(process_img,stream=True,classes=[target_class_index],conf=CONFIDENCE_THRESHOLD,verbose=False)
    except Exception as e:
        print(f"Error during YOLO inference:{e}")
        continue

    detections = np.empty((0,5))

    #process detections and format for tracker
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1_det,y1_det,x2_det,y2_det = box.xyxy[0]

            conf_det=math.ceil((box.conf[0]*100))/100
            cls_det = int(box.cls[0])

            if cls_det == target_class_index and conf_det>CONFIDENCE_THRESHOLD:
                currentArray=np.array([int(x1_det),int(y1_det),int(x2_det),int(y2_det),conf_det])
                detections=np.vstack((detections,currentArray))
            
        resultsTracker = tracker.update(detections)

            #drawing a line
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

        for result in resultsTracker:
            x1,y1,x2,y2,id = result
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            cx,cy = x1+w//2,y1+h//2

            # if limits[0]<cx<limits[2]:
            #     if limits[1]-10<cy<limits[1]+10:
            #         if id not in totalCount:
            #             totalCount.add(id)
            #             cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5) # Turn line greencvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0)) 

            line_x = limits[0]
            line_y_start = limits[1]
            line_y_end = limits[3]
            buffer=10

            if line_x - buffer < cx < line_x + buffer and line_y_start < cy < line_y_end:
                if id not in totalCount:
                    totalCount.add(id)
                    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)

       
            cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),scale=2, thickness=3, offset=10, colorR=(255,0,0), colorT=(255,255,255))
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)       
    
            cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50), scale=3, thickness=3, offset=10, colorR=(0,0,0), colorT=(255,255,255))
            cv2.imshow("Person Counter", img)
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processing stopped by user")
        break

cap.release()
cv2.destroyAllWindows()
print(f"Final Count:{len(totalCount)}")