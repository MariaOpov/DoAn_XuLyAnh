import os
import numpy as np
import scipy.io  
# Dòng 'import scipy.io' bị thiếu ở đây

# Thư viện:
# OS
# NUMPY
# SPICY

# Tình trạng: Hoàn thành
# Sửa lỗi:
# Kiểm tra file .mat và phân tích cấu trúc dữ liệu 'bb'.
    
# Lỗi ban đầu: Dùng "scipy.io.loadmat(file_path)" trực tiếp.
# Vấn đề: Giống như cv2.imread, cách này có thể không ổn định
    
# Sửa lỗi: Sử dụng phương pháp "an toàn" hơn:
# 1. Mở file bằng "with open(file_path, 'rb')" (read binary).
# 2. Tải dữ liệu từ file handle (f) thay vì từ đường dẫn:
# "scipy.io.loadmat(f)".

def check_mat(file_path): 
    print(f"Kiểm tra file: {file_path}")
    
    try:
        # LỖI: Dùng loadmat trực tiếp, không an toàn
        # Gây lỗi khi đường dẫn có Unicode hoặc quá phức tạp
        # mat_data = scipy.io.loadmat(file_path)
        with open(file_path, 'rb') as f:
            mat_data = scipy.io.loadmat(f)
            
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file tại {file_path}.")
        print("Hãy đảm bảo thư mục 'Baggages' nằm cùng cấp với file .py.")
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file .mat: {e}")
        return

    print("\n--- Keys trong file .mat ---")
    keys = mat_data.keys()
    for key in keys:
        print(f"* {key}")
    
    target_key = 'bb'
    
    if target_key in mat_data:
        bb_data = mat_data[target_key]
        
        print(f"\n--- Phân tích biến '{target_key}' ---")
        print(f"Biến chứa Bounding Box là: '{target_key}'")
        print(f"Kích thước của dữ liệu 'bb': {bb_data.shape}")
        
        num_rows_to_print = min(5, bb_data.shape[0])
        print(f"In ra {num_rows_to_print} dòng dữ liệu đầu tiên:")
        
        # Phần in bảng
        header = "| STT | img_id | xmin | xmax | ymin | ymax |"
        separator = "+-----+--------+------+------+------+------+"
        print(separator)
        print(header)
        print(separator)
        
        for i in range(num_rows_to_print):
            row = bb_data[i]
            # (Xác nhận 5 cột là [id, xmin, xmax, ymin, ymax])
            print(f"| {i+1:<3} | {int(row[0]):<6} | {int(row[1]):<4} | {int(row[2]):<4} | {int(row[3]):<4} | {int(row[4]):<4} |")
        
        print(separator)
        
        print("\n--- Xác nhận 5 cột dữ liệu ---")
        print("Cấu trúc (N, 5) được xác nhận là:")
        print("1. **img_id** (ID ảnh)")
        print("2. **xmin** (Tọa độ X nhỏ nhất)")
        print("3. **xmax** (Tọa độ X lớn nhất)")
        print("4. **ymin** (Tọa độ Y nhỏ nhất)")
        print("5. **ymax** (Tọa độ Y lớn nhất)")
        
    else:
        print(f"\nLỖI: Không tìm thấy biến '{target_key}' trong file .mat.")

if __name__ == "__main__":
    # Code này không có đường dẫn, nó tìm file "BoundingBox.mat"
    # ngay tại nơi bạn chạy lệnh, dẫn đến FileNotFoundError.
    # path_to_check = 'BoundingBox.mat'
    # check_mat(path_to_check)
    path_to_check = os.path.join('DoAn_XuLyAnh', 'Baggages', 'B0001', 'BoundingBox.mat')
    check_mat(path_to_check)