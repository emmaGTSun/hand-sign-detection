import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300
counter = 0
folder = "Data/A"

# 设置时间间隔为5秒
interval = 3
last_saved_time = time.time()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if hands:
        # Calculate combined bounding box coordinates
        min_x = min([hand["bbox"][0] for hand in hands])
        min_y = min([hand["bbox"][1] for hand in hands])
        max_x = max([hand["bbox"][0] + hand["bbox"][2] for hand in hands])
        max_y = max([hand["bbox"][1] + hand["bbox"][3] for hand in hands])

        w = max_x - min_x
        h = max_y - min_y

        aspectRatio = h / w

        # Check if the aspect ratio is within acceptable bounds
        if 0.5 <= aspectRatio <= 2:
            imgCrop = img[min_y - offset:max_y + offset, min_x - offset:max_x + offset]

            if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
                print("Invalid crop dimensions:", imgCrop.shape)
                continue

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

            # Draw a bounding box around both hands
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        else:
            cv2.putText(img, "Hands too close or too far apart!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # 检查是否达到指定的时间间隔
    current_time = time.time()
    if current_time - last_saved_time > interval:
        counter += 1
        cv2.imwrite(f'{folder}/Image_{counter}.jpg', imgWhite)
        print(f"Saved: Image_{counter}.jpg")
        last_saved_time = current_time

    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
