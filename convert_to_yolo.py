import cv2
import numpy as np
import scipy.io
import os
import shutil
import yaml

BASE_DIR = "XRay_Dataset"
IMAGE_SOURCE_DIR = "."
MAT_FILE = "BoundingBox.mat"
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

    images_train_dir = os.path.join(BASE_DIR, "images", "train")
    labels_train_dir = os.path.join(BASE_DIR, "labels", "train")
    
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    print(f"Đã tạo cấu trúc thư mục: {BASE_DIR}/...")

    try:
        mat_data = scipy.io.loadmat(MAT_FILE)
        bb_data = mat_data['bb']
        print(f"Đã đọc file {MAT_FILE}. Tổng số bounding box: {len(bb_data)}")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {MAT_FILE}.")
        return
    except Exception as e:
        print(f"LỖI khi đọc file .mat: {e}")
        return

    boxes_by_image = {}
    for row in bb_data:
        img_id = int(row[0])
        coords = row[1:].astype(int) 
        
        if img_id not in boxes_by_image:
            boxes_by_image[img_id] = []
        boxes_by_image[img_id].append(coords)

    total_images = len(boxes_by_image)
    print(f"Tổng số ảnh cần xử lý: {total_images}")

    processed_count = 0
    
    for img_id, boxes in boxes_by_image.items():
        img_filename = f"{img_id}.png"
        img_path_source = os.path.join(IMAGE_SOURCE_DIR, img_filename)
        label_filename = f"{img_id}.txt"
        label_path_dest = os.path.join(labels_train_dir, label_filename)

        img = cv2.imread(img_path_source)
        if img is None:
            print(f"CẢNH BÁO: Bỏ qua ảnh {img_filename} (Không tìm thấy).")
            continue
            
        H, W, _ = img.shape

        yolo_labels = []
        for xmin, xmax, ymin, ymax in boxes:
            yolo_line = convert_to_yolo_format(xmin, xmax, ymin, ymax, W, H)
            if yolo_line:
                yolo_labels.append(yolo_line)

        if yolo_labels:
            with open(label_path_dest, 'w') as f:
                f.write('\n'.join(yolo_labels) + '\n')
            
            img_path_dest = os.path.join(images_train_dir, img_filename)
            shutil.copy(img_path_source, img_path_dest)
            
            processed_count += 1
        
        if processed_count % 100 == 0 and processed_count > 0:
            print(f"Đã xử lý và lưu {processed_count} ảnh/nhãn.")

    print(f"Hoàn tất! Đã tạo {processed_count} file nhãn và di chuyển ảnh vào thư mục {BASE_DIR}.")

def create_yaml_config(base_path, class_name):
    config = {
        'path': base_path,
        'train': 'images/train',
        'val': 'images/train',
        'nc': 1,
        'names': {0: class_name}
    }
    
    yaml_path = "xray_config.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"\nĐã tạo file cấu hình YOLO: **{yaml_path}**")

if __name__ == "__main__":
    convert_to_yolo()
    
    create_yaml_config(
        base_path="./XRay_Dataset", 
        class_name="prohibited_item"
    )