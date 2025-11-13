from ultralytics import YOLO
import os

# Cần hoàn thành convert_to_yolo.py
# Yêu cầu có: convert_to_yolo.py, xray_config.yaml
# Đây là file dạy cho AI
# YOLO: You Only Look Once
# Nhiệm vụ:
# Tải mô hình YOLOv8 gốc. Đây là mô hình đã được huấn luyện trên dữ liệu lớn, nhưng chưa biết gì về ảnh X-ray.
# Đọc file xray_config.yaml (đường dẫn đến images/train và labels/train).
# Gọi hàm model.train() và chỉ định các thông số quan trọng (như epochs=50, imgsz=640). Đây là lúc máy tính bắt đầu xem ảnh xray của bạn lặp đi lặp lại để học cách nhận diện vật cấm.
# Sau khi huấn luyện xong, file này sẽ tự động lưu kết quả. Sản phẩm cuối cùng và quan trọng nhất của nó là file best.pt trong thư mục runs/detect/train/.

# Yêu cầu thực hiện: 
# Viết code để tải mô hình YOLOv8 gốc 
# Trỏ script của bạn đến file xray_config.yaml.
# Thiết lập thông số (epochs, imgsz, batch)
# Chạy huấn luyện: Gọi hàm model.train() và theo dõi quá trình.
