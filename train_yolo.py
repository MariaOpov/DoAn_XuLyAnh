import os
from ultralytics import YOLO  

# Đã import 'ultralogger' thay vì 'ultralytics'.
# Đã sửa lại tên thư viện thành "from ultralytics import YOLO".
#
# Khi chạy, code bị lỗi
# "OMP: Error #15: Initializing libiomp5md.dll...".
# Vấn đề: Đây là lỗi xung đột phổ biến trên Windows giữa PyTorch (YOLO)
# và các thư viện khác (như OpenCV, Numpy).
# Sửa lỗi 2: Thêm dòng 'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"'
# ngay bên dưới các dòng import để "bỏ qua" lỗi này và cho phép
# chương trình chạy tiếp.

# Tình trạng: đã hoàn thành

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train_model():
    """
    Hàm chính để huấn luyện mô hình YOLOv8.
    """
    print("=" * 70)
    print("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN (train_yolo.py)")
    print("=" * 70)

    # Xác định đường dẫn 
    try:
        # Lấy đường dẫn thư mục chứa code (ví dụ: D:\XLA_1\DoAn_XuLyAnh)
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.path.join(os.getcwd(), "DoAn_XuLyAnh")

    config_path = os.path.join(script_dir, "xray_config.yaml")
    
    if not os.path.exists(config_path):
        print(f"LỖI: Không tìm thấy file {config_path}")
        print("Hãy đảm bảo file 'convert_to_yolo.py' đã chạy thành công.")
        return

    print(f"Sử dụng file cấu hình: {config_path}")

    # Tải mô hình YOLOv8 
    # 'yolov8n.pt' là mô hình nano nhỏ nhất và nhanh nhất.
    # Lần đầu chạy, nó sẽ tự động tải file này về.
    print("Đang tải mô hình YOLOv8 ('yolov8n.pt')...")
    model = YOLO('yolov8n.pt')
    print("Tải mô hình gốc thành công.")

    # Bắt đầu huấn luyện 
    print("\nBắt đầu huấn luyện...")
    try:
      # Đường dẫn đến file .yaml
      # Số lần lặp lại toàn bộ dataset (50 là đủ)
      # Resize ảnh về 640x640
      # Xử lý 4 ảnh 1 lần (vì dataset nhỏ)
      # Lưu kết quả vào thư mục code (DoAn_XuLyAnh)
      # Đặt tên thư mục con là 'runs/train'
      # Moi epoch dai dien cho 1 lan huan luyen, AI chi xem anh mot lan duy nhat
        results = model.train(
            data=config_path,   
            epochs=50,          
            imgsz=640,          
            batch=4,            
            project=script_dir, 
            name="runs/train"  
        )
        
        print("\n" + "=" * 70)
        print("HUẤN LUYỆN HOÀN TẤT!")
        print("Mô hình đã được lưu trong thư mục 'DoAn_XuLyAnh/runs/train/weights/best.pt'")
        print("=" * 70)

    except Exception as e:
        print(f"\nĐÃ XẢY RA LỖI TRONG QUÁ TRÌNH HUẤN LUYỆN: {e}")
        print("Hãy kiểm tra lại đường dẫn trong file 'xray_config.yaml' và thư mục XRay_Dataset.")

if __name__ == "__main__":
    train_model()