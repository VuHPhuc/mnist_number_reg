import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Tải model đã được huấn luyện từ tệp 'chuviet.model'
model = tf.keras.models.load_model('chuviet.model')

# Khởi tạo biến đếm số thứ tự ảnh
image_number = 0

# Vòng lặp để đọc và dự đoán các tệp ảnh số đánh số từ 'num0.png' trở đi
while os.path.isfile(f"images/num{image_number}.png"):
    try: 
        # Đọc ảnh từ tệp ảnh
        img = cv2.imread(f"images/num{image_number}.png")[:, :, 0]
        
        # Chuyển đổi ảnh sang ảnh xám và đảo ngược giá trị để phù hợp với mô hình
        img = np.invert(np.array([img]))
        
        # Dự đoán kết quả của ảnh bằng mô hình đã tải
        prediction = model.predict(img)
        
        # Tạo danh sách các chữ số từ 0 đến 9
        english_letters = [str(int) for int in range(0, 10)]
        
        # In kết quả dự đoán và hiển thị ảnh
        print(f"Dự đoán có thể là: {english_letters[np.argmax(prediction)]}")
        plt.xlabel(f"Dự đoán có thể là: {english_letters[np.argmax(prediction)]}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        # Xử lý ngoại lệ nếu có lỗi xảy ra trong quá trình đọc hoặc dự đoán
        print("lỗi!")
    finally:
        # Tăng số thứ tự ảnh để đọc tiếp theo
        image_number += 1