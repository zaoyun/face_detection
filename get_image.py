import cv2
import os

# 定义数据集路径
data_dir = r"E:\Python_project\face_detection\pythonProject\.venv\dataset"
host_name = "host"
host_dir = os.path.join(data_dir, host_name)

if not os.path.exists(host_dir):
    os.makedirs(host_dir)

# 打开摄像头
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
while count < 800:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))
        file_path = os.path.join(host_dir, f"host_face_{count}.jpg")
        cv2.imwrite(file_path, face)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"Image {count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Capturing Host Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count >= 800:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} host face images successfully.")