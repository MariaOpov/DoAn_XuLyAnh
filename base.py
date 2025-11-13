import cv2
import numpy as np
import scipy.io
import os

# Môi trường sử dụng:
# OpenCV
# Scipy (cái này để đọc file đuôi mat)
# Ultralytics (để huấn luyện AI)

# Link tải GDXray:
# https://github.com/computervision-xray-testing/GDXray
# Chọn Baggage và tải, xong giải nén vào folder này

# Yêu cầu thực hiện:
# Thực hiện code demo cho tiền xử lý
# demo CLAHE để cho thấy thấy ảnh XRAY rõ nét
# demo phép toán Morphology để dọn dẹp nhiễu

