import cv2
import numpy as np

# 加载图像并转换为灰度图像
image = cv2.imread('1.webp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_orignal = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 加载人脸和眼睛检测器模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 检测人脸
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# 如果检测到了人脸
if len(faces) > 0:
    for (fx, fy, fw, fh) in faces:
        # 在人脸区域内进行皱纹检测
        face_roi = gray[fy:fy+int(3*fh/5), fx+int(2*fw/11):fx+int(9*fw/11)]

        # 加载眼睛检测器模型
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)

        # 绘制矩形框选区域并获取眼睛和嘴巴检测区域
        wrinkle_region = np.ones_like(face_roi) * 255
        for (ex, ey, ew, eh) in eyes:
            # 绘制矩形框选区域（眼睛上方）
            cv2.rectangle(wrinkle_region, (ex, ey + int(6 * eh / 9)), (ex + ew, ey - int(3 * eh / 9)), (0, 0, 0), -1)

        # 使用Sobel算子进行边缘检测
        sobel_x = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Sobel边缘图像
        wrinkle_image = np.uint8(gradient_magnitude > 100) * 255

        # 在皱纹检测区域之外同时进行皱纹检测
        wrinkle_image = cv2.bitwise_and(wrinkle_image, wrinkle_region)

        # 寻找检测到的皱纹区域的轮廓
        contours, _ = cv2.findContours(wrinkle_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在原图上绘制检测到的皱纹区域
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            #cv2.rectangle(face_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 对皱纹区域进行高斯滤波
            face_roi[y:y+h, x:x+w] = cv2.GaussianBlur(face_roi[y:y+h, x:x+w], (21, 21), 0)
        # 将处理后的人脸区域放回原图中
        gray[fy:fy+int(3*fh/5), fx+int(2*fw/11):fx+int(9*fw/11)] = face_roi
else:
    # 如果未检测到人脸，则在整个图像区域进行皱纹检测
    # 绘制矩形框选区域并获取眼睛检测区域
    wrinkle_region = np.ones_like(gray) * 255
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        # 绘制矩形框选区域（眼睛上方）
        cv2.rectangle(wrinkle_region, (ex, ey + int(6 * eh / 9)), (ex + ew, ey - int(3 * eh / 9)), (0, 0, 0), -1)

    # 使用Sobel算子进行边缘检测
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Sobel边缘图像
    wrinkle_image = np.uint8(gradient_magnitude > 150) * 255

    # 在眼睛之外同时进行皱纹检测
    wrinkle_image = cv2.bitwise_and(wrinkle_image, wrinkle_region)

    # 寻找检测到的皱纹区域的轮廓
    contours, _ = cv2.findContours(wrinkle_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原图上绘制检测到的皱纹区域
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # 对皱纹区域进行高斯滤波
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        gray[y:y+h, x:x+w] = cv2.GaussianBlur(gray[y:y+h, x:x+w], (15, 15), 0)

# 显示结果图像
cv2.imshow('image', image_orignal)
cv2.imshow('Wrinkle Detection', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

