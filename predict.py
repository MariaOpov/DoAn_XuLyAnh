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

# Yêu cầu thực hiện:
# Tải mô hình đã huấn luyện: Viết code để tải chính xác file YOLO
# Chạy dự đoán: Gọi hàm model.predict() trên ảnh Xray
# Xử lý kết quả: Lấy kết quả (tọa độ hộp, tên nhãn "prohibited_item", độ tự tin).

# Lỗi 1: Code ban đầu tìm file 'best.pt' trong thư mục
# 'runs/detect/train...' (đây là thư mục mặc định của YOLO cho 'predict').
# Vấn đề: File 'train_yolo.py' của chúng ta lưu mô hình ở 'runs/train...'.
# Sửa lỗi 1: Đã sửa lại logic tìm đường dẫn (xóa "detect") để tìm
# chính xác thư mục 'runs/train' (hoặc train2, train3...) mới nhất.
# Tình trạng: đã sửa

# Lỗi 2: Các phiên bản đầu tiên dùng 'show=True'
# của YOLO, khiến cửa sổ hiển thị bị tắt ngay lập tức.
# Sửa lỗi 2: Đã thay thế bằng 'res.plot()' (để lấy ảnh đã vẽ) và
# 'cv2.imshow()' kết hợp 'cv2.waitKey(0)' để giữ cửa sổ
# hiển thị cho đến khi người dùng nhấn phím.
# Tình trạng: đã sửa

# Tình trạng: đã hoàn thành
# Kết quả như mong muốn yêu cầu đề bài
def predict_image():
    """
    Hàm chính để chạy dự đoán trên một ảnh.
    """
    print("=" * 70)
    print("BẮT ĐẦU QUÁ TRÌNH DỰ ĐOÁN (predict.py)")
    print("=" * 70)

    # --- 1. Xác định đường dẫn ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.path.join(os.getcwd(), "DoAn_XuLyAnh")

    runs_dir = os.path.join(script_dir, "runs") 
    latest_train_dir = ""
    if os.path.exists(runs_dir):
        train_folders = sorted(
            [f for f in os.listdir(runs_dir) if f.startswith("train") and os.path.isdir(os.path.join(runs_dir, f))],
            reverse=True
        )
        if train_folders:
            latest_train_dir = os.path.join(runs_dir, train_folders[0])
            
    model_path = os.path.join(latest_train_dir, "weights", "best.pt")

    if not os.path.exists(model_path):
        print(f"LỖI: Không tìm thấy file 'best.pt' tại: {model_path}")
        return
    # Sau khi huan luyen AI, ta co the test anh bang cach cho anh ma AI da duoc huan luyen tu truoc
    # Nhu duong dan duoi
    image_to_test = os.path.join(script_dir, "Baggages", "B0001", "B0001_0001.png")
    
    if not os.path.exists(image_to_test):
        print(f"LỖI: Không tìm thấy ảnh kiểm tra tại: {image_to_test}")
        return
        
    print(f"\nĐã tải mô hình: {model_path}")
    print(f"Đang dự đoán trên ảnh cố định: {image_to_test}")

    try:
        model = YOLO(model_path)
        
        results = model.predict(source=image_to_test, save=True, conf=0.5) 
        
        res = results[0]
        annotated_image = res.plot() 
        
        print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
        num_objects = len(res.boxes)
        print(f"Tìm thấy {num_objects} vật thể.")
        
        if num_objects > 0:
            for box in res.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                print(f" - Vật thể: {class_name} (Độ tự tin: {confidence:.2f})")
        else:
            print("KHÔNG TÌM THẤY VẬT THỂ NÀO (với độ tự tin >= 0.5)")

        # Hiển thị kết quả (cv2.waitKey) 
        window_name = 'KET QUA DU DOAN (Nhan phim bat ky de thoat)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 800)
        
        cv2.imshow(window_name, annotated_image)
        
        print("\nĐang hiển thị ảnh. Nhấn phím bất kỳ để đóng cửa sổ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Đã đóng cửa sổ.")

    except Exception as e:
        print(f"ĐÃ XẢY RA LỖI KHI DỰ ĐOÁN: {e}")

if __name__ == "__main__":
    predict_image()