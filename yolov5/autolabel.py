from pathlib import Path
import torch
import cv2
import time
import os

# Parameters setup
CONFIDENCE      = 0.3
CROPPADDING     = 0
SAVECROPNAME    = "id10"
PATH            = "runs\mycropped"

CROPIDX         = int(input("\n> Continue save crop with index : "))
# Load model and custom weight
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'weight\weight_v1.pt', force_reload=True)
# Config camera or image
# cap = cv2.VideoCapture(r'example\example081.jpg')
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
                    #  0 Front-camera
                    #  1 Back-camera
                    #  2 USB-camera

t = [0.0, 0.0, 0.0]
while cap.isOpened():
    rval, frame = cap.read()
    img = frame.copy()

    if rval == True:
        # Resizing image
        # frame = cv2.resize(frame,(640,640))

        t[0] = time.time()
        # Yolo object detection
        result = model(frame)
        labels = result.xyxyn[0][:,5]
        position = result.xyxyn[0][:,:5]
        t[1] = time.time()

        # Visualization
        n = len(labels); x1y1x2y2 = []
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            Bounding_box = position[i]
            if Bounding_box[4] >= CONFIDENCE:
                x1, y1, x2, y2 = int(Bounding_box[0]*x_shape), int(Bounding_box[1]*y_shape), int(Bounding_box[2]*x_shape), int(Bounding_box[3]*y_shape)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame, 'Pill', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                x1y1x2y2.append((x1, y1, x2, y2))
        
        # Save crop
        if cv2.waitKey(10) & 0xFF == 13: # press Enter
            for i in range(n):
                # (BUG) Read and record new index 
                # with open(os.path.join(PATH, "record.txt"), 'w+', encoding = 'utf-8') as f:
                #     record = f.readlines()
                #     if len(record)==0: print(record, "> Create record.txt"); idx = 0
                #     else: idx = int(record[0])
                #     FILE = SAVECROPNAME + '_' + f'{idx:4.0f}'.replace(' ','0') + '.jpg'
                #     f.write(str(idx + 1))
                # Crop the image
                FILE = SAVECROPNAME + '_' + f'{CROPIDX:4.0f}'.replace(' ','0') + '.jpg'
                CROPIDX += 1
                print("> Cropped", FILE, x1y1x2y2[i])
                image_crop = img[
                    x1y1x2y2[i][1]-CROPPADDING : x1y1x2y2[i][3]+CROPPADDING,
                    x1y1x2y2[i][0]-CROPPADDING : x1y1x2y2[i][2]+CROPPADDING
                ]
                image_path = os.path.join(PATH, FILE)
                cv2.imwrite(image_path, image_crop)
        # Show FPS
        # print(f"> Yolo v5 detecting ({t[1]-t[0]:.3f}s)")
        cv2.putText(frame, f"fps: {1/(t[1]-t[0]):.0f} ({t[1]-t[0]:.3f}s)", (x_shape-10, y_shape-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.imshow("Pill", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):  # 13 : Enter
                                            # ord('A') : A
        break
cap.release()
cv2.destroyAllWindows()