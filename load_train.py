from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

# 定义图像大小和批量大小
img_size = (200, 200)
batch_size = 32

# 使用 ImageDataGenerator 进行数据增强
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,  # 降低数据增强的幅度
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# 加载训练集和验证集
train_data = train_datagen.flow_from_directory(
    r'E:\Python_project\face_detection\pythonProject\.venv\dataset',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_data = train_datagen.flow_from_directory(
    r'E:\Python_project\face_detection\pythonProject\.venv\dataset',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 创建 CNN 模型，添加正则化项
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3), kernel_regularizer=l2(0.001)),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D(2, 2),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# 编译模型，使用较小的学习率
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 定义学习率调度器，逐步降低学习率
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# 训练模型
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=80,
    callbacks=[reduce_lr]
)

# 保存模型
model.save('E:/Python_project/face_detection/pythonProject/.venv/face_recognition_model.h5')
print("Model saved successfully with adjustments.")

# 绘制 loss 曲线
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()