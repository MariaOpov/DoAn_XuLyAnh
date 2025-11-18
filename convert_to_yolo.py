import cv2
import numpy as np
import scipy.io
import os
import shutil
import yaml

# GHI CHÚ SỬA LỖI 
# định dạng YOLO và tự động tạo file config.
#
# Code cũ dùng đường dẫn tương đối (relative path)
# như 'BoundingBox.mat' và '.', khiến code không tìm thấy file
# khi chạy từ thư mục gốc.
# Sửa lỗi 1: Đã thay thế bằng logic lấy đường dẫn TUYỆT ĐỐI
# (dùng os.path.abspath(__file__)) để luôn tìm đúng thư mục
# "Baggages" và "XRay_Dataset" bất kể chạy từ đâu.
#
# Code cũ giả định tên ảnh là '1.png', '2.png'.
# Đã sửa logic để tạo tên file chính xác 
# (ví dụ: 'B0001_0001.png') bằng f-string.
#
# Code cũ dùng 'cv2.imread()' trực tiếp.
# Sửa lỗi 3: Đã thay thế bằng phương pháp 'np.fromfile' và 'cv2.imdecode'
# để đọc file an toàn, tránh lỗi Unicode/Path.
#
# Code cũ đọc ảnh 3 kênh màu và dùng
# 'H, W, _ = img.shape', gây lỗi khi chúng ta đọc ảnh xám (grayscale).
# Sửa lỗi 4: Đã sửa lại logic đọc ảnh grayscale ('cv2.IMREAD_GRAYSCALE')
# và lấy kích thước 'H, W = img.shape[:2]'.

# Tình trạng: đã hoàn thành

# Lấy thư mục gốc của project
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR) 
except NameError:
    ROOT_DIR = os.getcwd() 
    SCRIPT_DIR = os.path.join(ROOT_DIR, "DoAn_XuLyAnh")

# Đường dẫn đến thư mục chứa code
CODE_DIR = SCRIPT_DIR

# Đường dẫn đến thư mục B0001 (nơi chứa ảnh và file .mat)
SOURCE_DATA_DIR = os.path.join(CODE_DIR, "Baggages", "B0001")

# Đường dẫn file .mat
MAT_FILE_PATH = os.path.join(SOURCE_DATA_DIR, "BoundingBox.mat")

# Thư mục đích cho dataset đã xử lý
BASE_DIR_DEST = os.path.join(CODE_DIR, "XRay_Dataset")

CLASS_INDEX = 0

def convert_to_yolo_format(xmin, xmax, ymin, ymax, W, H):
    if W == 0 or H == 0:
        return None
    
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    box_width = xmax - xmin
    box_height = ymax - ymin
    
    yolo_cx = center_x / W
    yolo_cy = center_y / H
    yolo_w = box_width / W
    yolo_h = box_height / H
    
    return f"{CLASS_INDEX} {yolo_cx:.6f} {yolo_cy:.6f} {yolo_w:.6f} {yolo_h:.6f}"

def convert_to_yolo():
    print("Bắt đầu quy trình chuyển đổi dữ liệu GDXray sang định dạng YOLO...")

    images_train_dir = os.path.join(BASE_DIR_DEST, "images", "train")
    labels_train_dir = os.path.join(BASE_DIR_DEST, "labels", "train")
    
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    print(f"Đã tạo cấu trúc thư mục: {BASE_DIR_DEST}/...")

    try:
        # Đọc file .mat an toàn 
        with open(MAT_FILE_PATH, 'rb') as f:
            mat_data = scipy.io.loadmat(f)
        bb_data = mat_data['bb']
        print(f"Đã đọc file {MAT_FILE_PATH}. Tổng số bounding box: {len(bb_data)}")
        
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {MAT_FILE_PATH}.")
        print("Hãy đảm bảo bạn đã chạy 'check_mat.py' và đường dẫn là đúng.")
        return
    except Exception as e:
        print(f"LỖI khi đọc file .mat: {e}")
        return

    # Logic nhóm theo image_id
    boxes_by_image = {}
    for row in bb_data:
        img_id = int(row[0])
        # (id, xmin, xmax, ymin, ymax)
        coords = row[1:].astype(float) 
        
        if img_id not in boxes_by_image:
            boxes_by_image[img_id] = []
        boxes_by_image[img_id].append(coords)

    total_images = len(boxes_by_image)
    print(f"Tổng số ảnh cần xử lý (từ B0001): {total_images}")

    processed_count = 0
    
    for img_id, boxes in boxes_by_image.items():
        
        # Sửa lại tên file ảnh cho đúng
        img_filename = f"B0001_{img_id:04d}.png" # Ví dụ: B0001_0001.png
        img_path_source = os.path.join(SOURCE_DATA_DIR, img_filename)
        
        label_filename = f"B0001_{img_id:04d}.txt" # Ví dụ: B0001_0001.txt
        label_path_dest = os.path.join(labels_train_dir, label_filename)

        # Đọc ảnh an toàn (grayscale)
        try:
            file_bytes = np.fromfile(img_path_source, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError("Không thể giải mã ảnh")
        except Exception as e:
            print(f"CẢNH BÁO: Bỏ qua ảnh {img_filename} (Lỗi đọc file: {e}).")
            continue
            
        # Lấy H, W từ ảnh grayscale 
        H, W = img.shape[:2]

        yolo_labels = []
        for xmin, xmax, ymin, ymax in boxes:
            yolo_line = convert_to_yolo_format(xmin, xmax, ymin, ymax, W, H)
            if yolo_line:
                yolo_labels.append(yolo_line)

        if yolo_labels:
            with open(label_path_dest, 'w') as f:
                f.write('\n'.join(yolo_labels) + '\n')
            
            # Copy ảnh gốc vào thư mục train
            img_path_dest = os.path.join(images_train_dir, img_filename)
            shutil.copy(img_path_source, img_path_dest)
            
            processed_count += 1
        
        if processed_count % 10 == 0 and processed_count > 0:
            print(f"Đã xử lý và lưu {processed_count}/{total_images} ảnh/nhãn.")

    print(f"Hoàn tất! Đã tạo {processed_count} file nhãn và di chuyển ảnh vào thư mục {BASE_DIR_DEST}.")

def create_yaml_config(dataset_abs_path, class_name):
    """Tạo file YAML với đường dẫn tuyệt đối"""
    config = {
        'path': dataset_abs_path, 
        'train': 'images/train',
        'val': 'images/train',
        'nc': 1,
        'names': {0: class_name}
    }
    
    # Lưu file .yaml vào thư mục code (DoAn_XuLyAnh)
    yaml_path = os.path.join(CODE_DIR, "xray_config.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"\nĐã tạo file cấu hình YOLO: {yaml_path}")
    print(f"Nội dung file YAML:\n{yaml.dump(config)}")

if __name__ == "__main__":
    # 1. Chuyển đổi dữ liệu
    convert_to_yolo()
    
    # 2. Tạo file config
    create_yaml_config(
        dataset_abs_path=BASE_DIR_DEST, 
        class_name="prohibited_item"
    )