import gradio as gr
import os
import cv2
import numpy as np
from ultralytics import YOLO

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
def get_best_model_path():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.path.join(os.getcwd(), "DoAn_XuLyAnh")

    runs_dir = os.path.join(script_dir, "runs")
    latest_train_dir = ""
    
    if os.path.exists(runs_dir):
        # Tìm thư mục train mới nhất
        train_folders = sorted(
            [f for f in os.listdir(runs_dir) if f.startswith("train") and os.path.isdir(os.path.join(runs_dir, f))],
            reverse=True
        )
        if train_folders:
            latest_train_dir = os.path.join(runs_dir, train_folders[0])
    
    model_path = os.path.join(latest_train_dir, "weights", "best.pt")
    
    if os.path.exists(model_path):
        return model_path
    else:
        return None

# Tải mô hình toàn cục (Global) để không phải load lại mỗi lần dự đoán
MODEL_PATH = get_best_model_path()
if MODEL_PATH:
    print(f"Đã tải mô hình từ: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
else:
    print("LỖI: Không tìm thấy file 'best.pt'.")
    model = None

# --- 2. HÀM DỰ ĐOÁN ---
def predict_xray(input_image):
    """
    Hàm này nhận ảnh từ Gradio (dạng numpy array), chạy YOLO,
    và trả về ảnh có vẽ bounding box.
    """
    if model is None:
        return input_image 
    
    # input_image từ Gradio là RGB, YOLO xử lý tốt cả RGB
    # Chạy dự đoán
    results = model.predict(source=input_image, conf=0.5)
    
    res = results[0]
    
    annotated_img_bgr = res.plot()
    
    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
    
    # Tạo thông báo kết quả
    num_objects = len(res.boxes)
    info_text = f"Tìm thấy {num_objects} vật cấm."
    
    return annotated_img_rgb, info_text

# --- 3. TẠO GIAO DIỆN GRADIO ---
# Tạo giao diện
demo = gr.Interface(
    fn=predict_xray,           
    inputs=gr.Image(label="Tải ảnh X-ray lên đây"), 
    outputs=[
        gr.Image(label="Kết quả phát hiện"),        
        gr.Textbox(label="Thông tin")               
    ],
    title="HỆ THỐNG PHÁT HIỆN VẬT CẤM X-RAY",
    description="Tải lên ảnh chụp X-ray hành lý để AI phát hiện vật cấm (Kéo, Dao, v.v.)",
    theme="default",
    allow_flagging="never"
)

# --- 4. CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    print("Đang khởi động Web App...")
    demo.launch(inbrowser=True)