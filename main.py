import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import math
import matplotlib.pyplot as plt

# Import các thư viện cần thiết

data = pd.read_csv('Data/train.csv')

# Đọc dữ liệu từ tệp csv

data = data.values

# Chuyển dữ liệu thành mảng numpy
#[:,1:] để lấy tất cả các hàng và tất cả các cột từ cột thứ hai trở đi (cột 1 đến cuối cùng),
#và data[:,0] để lấy tất cả các hàng và chỉ cột đầu tiên (cột 0). Kết quả là data_X chứa dữ liệu ảnh và data_Y chứa nhãn tương ứng.

data_X, data_Y = data[:,1:], data[:,0]

#Ở dòng này, chúng ta chia tất cả các giá trị trong data_X cho 255.0. Việc này được thực hiện để tiêu chuẩn hóa giá trị pixel trong khoảng từ 0 đến 1.
#Bằng cách chia tỷ lệ giá trị pixel theo giá trị tối đa (255),
#chúng ta đảm bảo rằng tất cả các giá trị pixel nằm trong khoảng từ 0 đến 1, giúp quá trình huấn luyện mô hình diễn ra tốt hơn.

data_X = data_X / 255.

# Chia dữ liệu thành hai phần: data_X là pixel và data_Y là kết quả (nhãn)
# Chia tỷ lệ giá trị pixel trong khoảng từ 0 đến 1

TRAIN_LIMIT = math.ceil(0.7 * data_X.shape[0])
VALID_LIMIT = TRAIN_LIMIT + math.ceil(0.2 * data_X.shape[0])

# Tính toán giới hạn cho mỗi phần dữ liệu (train, validation, test) dựa trên tỷ lệ phần trăm

train_X, val_X, test_X = data_X[:TRAIN_LIMIT], data_X[TRAIN_LIMIT:VALID_LIMIT], data_X[VALID_LIMIT:]
train_Y, val_Y, test_Y = data_Y[:TRAIN_LIMIT], data_Y[TRAIN_LIMIT:VALID_LIMIT], data_Y[VALID_LIMIT:]

# Chia dữ liệu thành các tập huấn luyện, validation và kiểm tra

model = Sequential([
    InputLayer(input_shape=(data_X.shape[1])),
    Dense(784, activation='relu'),
    Dense(10, activation='relu'),
# Ở lớp Dense dưới nếu nhận dạng chữ sẽ là 26 còn nhận dạng số sẽ là 10
    Dense(10, activation='softmax')
])

# Xây dựng mô hình Sequential với các lớp
# - Lớp InputLayer với kích thước đầu vào dựa trên số chiều của data_X
# - Lớp Dense với 784 nơ-ron và hàm kích hoạt ReLU
# - Lớp Dense thứ hai với 10 nơ-ron và hàm kích hoạt ReLU
# - Lớp Dense cuối cùng với 26 nơ-ron (cho chữ cái) và hàm kích hoạt softmax để tính phần trăm dự đoán

model.summary()

# Hiển thị thông tin về mô hình

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Compile mô hình với trình tối ưu hóa 'adam', hàm mất mát 'sparse_categorical_crossentropy' và độ đo 'accuracy'
# optimizer='adam': Sử dụng thuật toán tối ưu Adam để tối thiểu hóa hàm mất mát và điều chỉnh các trọng số trong mô hình.
# loss='sparse_categorical_crossentropy': Sử dụng hàm mất mát loại "sparse categorical cross-entropy" 
# để đo lường sai số giữa đầu ra dự đoán và nhãn thực tế. Đây là hàm mất mát phù hợp cho bài toán phân loại với các nhãn không được one-hot encoded.
# metrics=['accuracy']: Đánh giá hiệu suất của mô hình bằng độ chính xác (accuracy).

fitting = model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=3, batch_size=10)

# Thực hiện quá trình huấn luyện với dữ liệu huấn luyện, kiểm tra trên dữ liệu validation
# Số epochs là 3 và batch_size là 50

history = model.evaluate(test_X,test_Y)

# Đánh giá mô hình trên dữ liệu kiểm tra

model.save('model-batchsize-10.keras')

# Lưu mô hình đã huấn luyện vào tệp 'model-batchsize-50.keras'

acc_res = fitting.history['accuracy']
val_acc_res = fitting.history['val_accuracy']
loss_res = fitting.history['loss']
val_loss_res = fitting.history['val_loss']

# Trích xuất các thông số từ quá trình huấn luyện

plt.plot(range(len(acc_res)),acc_res, label='Accuracy')
plt.plot(range(len(val_acc_res)),val_acc_res, label='Val_Accuracy')
plt.legend()
plt.show()

# Vẽ đồ thị cho accuracy và val_accuracy

plt.plot(range(len(loss_res)),loss_res, label='Loss')
plt.plot(range(len(val_loss_res)),val_loss_res, label='Val_Loss')
plt.legend()
plt.show()

# Vẽ đồ thị cho loss và val_loss
