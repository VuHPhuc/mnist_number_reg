import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Tải mô hình đã huấn luyện từ tệp 'model-batchsize-50.keras'
model = keras.models.load_model('model-batchsize-10.keras')

# Đọc dữ liệu từ tệp 'train.csv' bằng pandas
data = pd.read_csv('Data/train.csv')

# Chọn cột từ index 1 trở đi và lấy giá trị của tất cả các hàng
data = data.values[:, 1:]

# Chọn chỉ số ảnh để kiểm tra
indexImg = 12421

# Dự đoán trên toàn bộ dữ liệu sử dụng mô hình đã tải
predictions = model.predict(data)

# Lấy ảnh kiểm tra từ dữ liệu theo chỉ số đã chọn
test_img = data[indexImg]

# Reshape ảnh kiểm tra thành kích thước (28, 28) để hiển thị
test_img = np.reshape(test_img, (28, 28))

# Hiển thị ảnh kiểm tra
plt.imshow(test_img)

# Hiển thị nhãn dự đoán tương ứng với ảnh
predicted_label = np.argmax(predictions[indexImg])
print(f"Dự đoán có thể là: {predicted_label}")
plt.xlabel(f'Dự Đoán Có Thể Là: {predicted_label}')

# Hiển thị đồ thị
plt.show()