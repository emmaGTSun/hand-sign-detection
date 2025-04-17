import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("/Users/sunguangtian/Desktop/HandSignDectection/Model/keras_model.h5",
                        "/Users/sunguangtian/Desktop/HandSignDectection/Model/labels.txt")
offset = 20
imgSize = 300

labels = ["A", "B", "C"]
targetLetter = None  # 这将是我们的目标字母
score = 0
lastScore = 0  # 用来确保我们只为每个手势增加一次分数

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image. Check your camera.")
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    # 创建一个白色背景的图像用于分类
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if hands:
        min_x = min([hand["bbox"][0] for hand in hands])
        min_y = min([hand["bbox"][1] for hand in hands])
        max_x = max([hand["bbox"][0] + hand["bbox"][2] for hand in hands])
        max_y = max([hand["bbox"][1] + hand["bbox"][3] for hand in hands])

        w = max_x - min_x
        h = max_y - min_y

        aspectRatio = h / w

        cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2  # 手势中心点

        if 0.5 <= aspectRatio <= 2:  # 确保我们的手势比例是可以接受的
            imgCrop = img[min_y - offset:max_y + offset, min_x - offset:max_x + offset]

            if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
                print("Invalid crop dimensions:", imgCrop.shape)
                continue

            # 根据手势的纵横比调整图像大小
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

            prediction, index = classifier.getPrediction(imgWhite)

            if targetLetter == labels[index] and score == lastScore:
                score += 10
                lastScore = score

            # 显示检测到的手势字母
            cv2.putText(imgOutput, labels[index], (cx, cy - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
            cv2.putText(imgOutput, f"Score: {score}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(img, "Hands too close or too far apart!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示目标字母
    cv2.putText(imgOutput, f"Target Letter: {targetLetter if targetLetter else 'None'}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    # 检查按下的键，然后设置目标字母
    if key in [ord("A"), ord("B"), ord("C")]:
        targetLetter = chr(key)

    if key == 27:  # 按下ESC键关闭
        break

cap.release()
cv2.destroyAllWindows()