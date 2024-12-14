import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 加载训练好的模型
model = load_model('E:/Python_project/face_detection/pythonProject/.venv/face_recognition_model.h5')

# 定义图像大小
img_size = (200, 200)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 加载人脸检测的 Haar 特征分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 类别标签（根据训练时的 class_indices 动态获取）
class_labels = {0: 'Host', 1: 'Intruder'}

# 设定预测置信度的阈值
confidence_threshold = 0.96

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, img_size)
        face = face / 255.0  # 归一化处理
        face = np.expand_dims(face, axis=0)  # 扩展维度以匹配模型输入

        # 预测人脸属于哪个类别
        prediction = model.predict(face)
        class_id = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][class_id]

        # 打印预测的置信度和类别索引
        print(f"Predicted class: {class_id}, Confidence: {confidence:.2f}")

        # 判断预测结果
        if confidence > confidence_threshold:
            label = class_labels[class_id]  # 获取标签
            # 绘制矩形框和标签
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Real-time Face Detection', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()