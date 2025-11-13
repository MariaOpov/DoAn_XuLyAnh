from ultralytics import YOLO
import os
import cv2 

# Đây là bộ não chính của bài tập
# Cần hoàn thành train_yolo.py
# Yêu cầu có: best.pt 
# Tải mô hình best.pt (là sản phẩm do file train_yolo.py tạo ra).
# Chỉ định đường dẫn đến một ảnh X-ray bất kỳ mà bạn muốn thử nghiệm
# Dùng mô hình đã tải để "nhìn" bức ảnh và tìm vật thể.
# Vẽ một hình chữ nhật và nhãn lên các vật thể mà nó tìm thấy, sau đó hiển thị kết quả đó lên màn hình.
