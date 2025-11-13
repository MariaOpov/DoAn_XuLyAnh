import cv2
import numpy as np
import scipy.io
import os

# Cần hoàn thành check_mat.py
# Sau check_mat.py, giúp ta hiểu được dữ liệu (bb chứa tọa độ img_id, xmin, xmax, ymin, ymax)
# Nhiệm vụ của convert_to_yolo sẽ là:
# Tải toàn bộ đáp án từ BoundingBox.mat
# Đọc file ảnh .png tương ứng để lấy kích thước (chiều rộng, chiều cao) thật của ảnh.
# Thực hiện phép toán để chuyển đổi tọa độ pixel sang định dạng YOLO 
# Lưu kết quả đã chuyển đổi vào một file .txt mới
# Nói chung, Nó tạo ra dữ liệu sạch (các file .txt) để chuẩn bị cho mô hình AI (YOLO).

# Yêu cầu thực hiện:
# Dựa trên check_mat.py, viết hàm chuyển đổi pixel sang định dạng YOLO
# Chạy script để tự động tạo ra các file .txt tương ứng cho mỗi file .png
# Tạo cấu trúc thư mục XRay_Dataset (với images/train và labels/train)
# Copy các file ảnh .png và các file nhãn .txt (vừa tạo) vào đúng vị trí.
# Tạo file xray_config.yaml.
# Viết cấu hình trỏ đường dẫn chính xác đến thư mục XRay_Dataset và định nghĩa tên lớp (0: prohibited_item).

# Sau khi hoàn thành convert_to_yolo.py, ta sẽ tạo xray_config.yaml
# File này là file cấu hình cho YOLO
# Dữ liệu ở đâu: Nó trỏ đến đường dẫn của thư mục XRay_Dataset.
# Ảnh huấn luyện ở đâu: Nó chỉ đến thư mục images/train.
# Nhãn huấn luyện ở đâu: Nó chỉ đến thư mục labels/train.
# Nó định nghĩa tên của các lớp (class). Ví dụ: 0: prohibited_item.