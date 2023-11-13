import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Tải model từ tệp 'model-batchsize-50.keras'
model = keras.models.load_model('model-batchsize-50.keras')

# Tạo danh sách các chữ cái tiếng Anh từ a đến z 
# Nhận Dạng chữ số sẽ không dùng cái này => (ctrl + /) 
english_letters = []
for letter in range(97, 123):
    english_letters.append(chr(letter))
    
# Đọc dữ liệu từ tệp 'a_z_train.csv' hoặc 'train.csv' bằng pandas và chuyển đổi thành mảng numpy
data =  pd.read_csv('Data/a_z_train.csv')
data = data.values[:, 0:data.shape[1]-1]

# Chọn chỉ số ảnh để kiểm tra
indexImg = 123474

# Dự đoán trên toàn bộ dữ liệu sử dụng mô hình đã tải
predictions = model.predict(data)

# Lấy ảnh kiểm tra từ dữ liệu
test_img = data[indexImg]
test_img = np.reshape(test_img, (28, 28))


print(f"Dự đoán có thể là: {english_letters[np.argmax(predictions[indexImg])]}")
# Hiển thị ảnh và kết quả dự đoán
plt.imshow(test_img)
# Xoá English_letters nếu nhận dạng số
plt.xlabel(f'Dự đoán có thể là: {english_letters[np.argmax(predictions[indexImg])]}')
plt.show()