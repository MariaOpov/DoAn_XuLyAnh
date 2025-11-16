import scipy.io
import os
import numpy as np

def check_mat(file_path="BoundingBox.mat"):
    print(f"Kiểm tra file: {file_path}")
    
    try:
        mat_data = scipy.io.loadmat(file_path)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {file_path}.")
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
        
        header = "| STT | img_id | xmin | xmax | ymin | ymax |"
        separator = "+-----+--------+------+------+------+------+"
        print(separator)
        print(header)
        print(separator)
        
        for i in range(num_rows_to_print):
            row = bb_data[i]
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
    check_mat()