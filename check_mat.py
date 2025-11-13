import scipy.io
import os
import numpy as np

# Nhiệm vụ của check_mat là đọc và giải mã BoundingBox.mat của bộ dữ liệu GDXray đã note ở base
# File kiểm tra: BoundingBox.mat
# Mục tiêu cụ thể của check_mat bao gồm:
# In ra các biến bên trong file .mat (__header__, __version__, bb) để tìm dữ liệu đáp án được lưu ở đâu
# Sau khi tìm ra biến chứa đáp án (bb), nó in kích thước (.shape) và vài dòng dữ liệu
# Sau cùng là nhìn vào dữ liệu thô, từ đó ta sẽ suy luận chính xác năm cột dữ liệu: img_id, xmin, xmax, ymin, ymax
# Nói ngắn gọn, file này giúp chúng ta hiểu được dữ liệu gốc trước khi xử lý nó.

