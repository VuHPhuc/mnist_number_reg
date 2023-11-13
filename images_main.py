import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Tải dữ liệu MNIST từ keras.datasets.mnist
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Tiêu chuẩn hóa dữ liệu bằng cách chia tỷ lệ giá trị pixel trong khoảng từ 0 đến 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Xây dựng mô hình mạng neural
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Lớp nhập dữ liệu và làm phẳng hình ảnh 28x28 thành vector 1D có kích thước 784
model.add(tf.keras.layers.Dense(128, activation='relu'))  # Lớp fully connected với 128 nơ-ron và hàm kích hoạt ReLU
model.add(tf.keras.layers.Dense(128, activation='relu'))  # Lớp fully connected thứ hai với 128 nơ-ron và hàm kích hoạt ReLU
model.add(tf.keras.layers.Dense(10, activation='softmax'))  # Lớp đầu ra với 10 nơ-ron và hàm kích hoạt Softmax để dự đoán xác suất cho từng lớp

# Biên dịch mô hình bằng cách chọn optimizer 'adam', loss function 'sparse_categorical_crossentropy' và metrics là 'accuracy'
# optimizer='adam': Sử dụng thuật toán tối ưu Adam để tối thiểu hóa hàm mất mát và điều chỉnh các trọng số trong mô hình.
# loss='sparse_categorical_crossentropy': Sử dụng hàm mất mát loại "sparse categorical cross-entropy" 
# để đo lường sai số giữa đầu ra dự đoán và nhãn thực tế. Đây là hàm mất mát phù hợp cho bài toán phân loại với các nhãn không được one-hot encoded.
# metrics=['accuracy']: Đánh giá hiệu suất của mô hình bằng độ chính xác (accuracy).
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình trên tập huấn luyện với số epochs là 3
model.fit(x_train, y_train, epochs=3)

# Lưu mô hình đã huấn luyện vào tệp 'chuviet.model'
model.save('chuviet.model')