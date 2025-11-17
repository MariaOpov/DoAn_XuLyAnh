import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

def print_f(message):
    """Hàm in thông báo thay thế cho print"""
    print(message)


# Môi trường sử dụng:
# OpenCV
# Scipy (cái này để đọc file đuôi mat)
# Ultralytics (để huấn luyện AI)

# Thư viện:
# CV2
# NUMPY
# OS
# MATPLOTLIB
# RANDOM

# Link tải GDXray:
# https://github.com/computervision-xray-testing/GDXray
# Chọn Baggage và tải, xong giải nén vào folder này

# Yêu cầu thực hiện:
# Thực hiện code demo cho tiền xử lý
# demo CLAHE để cho thấy thấy ảnh XRAY rõ nét
# demo phép toán Morphology để dọn dẹp nhiễu

# Tình trạng: Hoàn thành
# Sửa lỗi:

# Kiểm tra ảnh PNG trong thư mục Baggages
    
# --- GHI CHÚ SỬA LỖI ---
    # Lỗi ban đầu: Code tìm thư mục "GDXray" (tên trong tài liệu)
    # Sửa lỗi 1: Đổi tên thư mục tìm kiếm thành "Baggages" (tên thư mục thực tế).
    
    # Lỗi thứ hai: Code không tìm thấy thư mục "Baggages" do lỗi đường dẫn
    # tương đối (relative path) khi chạy từ các thư mục khác nhau.
    
    # Sửa lỗi 2: Sử dụng logic "try/except" để xử lý cả 2 trường hợp:
    # Thứ nhất: try: Lấy đường dẫn tuyệt đối (an toàn nhất) dựa trên vị trí
    #      của file .py này (khi chạy bằng lệnh `python .../base.py`).
    # Thứ hai: except: Dùng đường dẫn tương đối (an toàn thứ hai) 
    #      (khi chạy bằng nút 'Run')

def kiem_tra_gdxray_png():
    """Kiểm tra ảnh PNG trong thư mục Baggages"""
    print_f("KIEM TRA ANH PNG TRONG Baggages")
    
    try:
        # Lấy đường dẫn của file .py này (ví dụ: D:\XLA_1\DoAn_XuLyAnh\base.py)
        script_path = os.path.abspath(__file__)
        # Lấy thư mục chứa file .py (ví dụ: D:\XLA_1\DoAn_XuLyAnh)
        script_dir = os.path.dirname(script_path)
        # Nối với "Baggages" (ví dụ: D:\XLA_1\DoAn_XuLyAnh\Baggages)
        target_folder = os.path.join(script_dir, "Baggages") 
        
    except NameError:
        # Cần tìm: D:\XLA_1\DoAn_XuLyAnh\Baggages
        target_folder = os.path.join("DoAn_XuLyAnh", "Baggages") 
    
    
    if not os.path.exists(target_folder):
        print_f(f"LOI: Khong tim thay thu muc tai {target_folder}")
        print_f("Kiem tra lai duong dan hoac thu muc lam viec.")
        return False, []
    
    # Tìm tất cả ảnh PNG trong Baggages và các thư mục con
    danh_sach_anh = []
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Chỉ lấy ảnh, bỏ qua các file nhãn .mat
                if 'BoundingBox' not in file and 'ground_truth' not in file:
                    danh_sach_anh.append(os.path.join(root, file))
    
    if not danh_sach_anh:
        print_f(f"KHONG TIM THAY ANH TRONG {target_folder}")
        print_f(f"Kiem tra lai thu muc {target_folder} co chua anh khong")
        return False, []
    
    print_f(f"TIM THAY " + str(len(danh_sach_anh)) + f" ANH TRONG {target_folder}")
    return True, danh_sach_anh

def chon_anh_ngau_nhien(danh_sach_anh):
    """Chọn ngẫu nhiên một ảnh từ danh sách"""
    if not danh_sach_anh:
        return None
    
    anh_ngau_nhien = random.choice(danh_sach_anh)
    print_f("CHON ANH NGAU NHIEN: " + os.path.basename(anh_ngau_nhien))
    print_f("DUONG DAN: " + anh_ngau_nhien)
    return anh_ngau_nhien


def tai_anh_xray(duong_dan):
    """Tải ảnh X-ray và tối ưu hóa (Sử dụng phương pháp an toàn)"""
    try:
        print_f("DANG TAI ANH: " + os.path.basename(duong_dan))
        
        file_bytes = np.fromfile(duong_dan, dtype=np.uint8)
        anh = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if anh is None:
            print_f("LOI: Khong the doc anh")
            return None
            
        print_f("KICH THUOC ANH GOC: " + str(anh.shape))
        
        h, w = anh.shape
        if h > 2000 or w > 2000:
            scale = min(1000/h, 1000/w)
            new_h, new_w = int(h * scale), int(w * scale)
            anh = cv2.resize(anh, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print_f("DA THAY DOI KICH THUOC: " + str(anh.shape))
        
        # Chuẩn hóa ảnh để cải thiện độ tương phản
        if anh.dtype != np.uint8:
            anh = cv2.normalize(anh, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Áp dụng Gaussian blur nhẹ để giảm nhiễu trước khi xử lý
        anh = cv2.GaussianBlur(anh, (3, 3), 0)
        
        print_f("KICH THUOC ANH DA XU LY: " + str(anh.shape))
        return anh
        
    except Exception as e:
        print_f("LOI KHI DOC ANH: " + str(e))
        return None


def ap_dung_clahe_cai_tien(anh, clip_limit=3.0, grid_size=(16, 16)):
    """Áp dụng CLAHE cải tiến để tăng cường độ tương phản"""
    print_f("AP DUNG CLAHE CAI TIEN")
    print_f("Clip limit: " + str(clip_limit) + ", Grid size: " + str(grid_size))
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    anh_clahe = clahe.apply(anh)
    
    alpha = 1.2  
    beta = 10    
    anh_cai_tien = cv2.convertScaleAbs(anh_clahe, alpha=alpha, beta=beta)
    
    print_f("HOAN THANH CLAHE CAI TIEN")
    return anh_cai_tien

def ap_dung_morphology_cai_tien(anh, kernel_size=5):
    """Áp dụng Morphology cải tiến để khử nhiễu"""
    print_f("AP DUNG MORPHOLOGY CAI TIEN")
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    anh_opening = cv2.morphologyEx(anh, cv2.MORPH_OPEN, kernel)
    anh_closing = cv2.morphologyEx(anh, cv2.MORPH_CLOSE, kernel)
    anh_ket_hop = cv2.morphologyEx(anh_opening, cv2.MORPH_CLOSE, kernel)
    
    print_f("HOAN THANH MORPHOLOGY CAI TIEN")
    return anh_opening, anh_closing, anh_ket_hop

def lam_ro_anh_xray(anh):
    """Làm rõ ảnh X-ray bằng kỹ thuật nâng cao"""
    print_f("LAM RO ANH X-RAY")
    
    anh_clahe = ap_dung_clahe_cai_tien(anh, clip_limit=4.0, grid_size=(12, 12))
    anh_opening, anh_closing, anh_morphology = ap_dung_morphology_cai_tien(anh_clahe, kernel_size=3)
    
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    anh_sac_net = cv2.filter2D(anh_morphology, -1, kernel_sharpen)
    
    print_f("HOAN THANH LAM RO ANH X-RAY")
    return anh_clahe, anh_morphology, anh_sac_net

def hien_thi_ket_qua_ro_net(anh_goc, anh_clahe, anh_morphology, anh_ket_hop):
    """Hiển thị kết quả rõ nét với kích thước tối ưu"""
    print_f("HIEN THI KET QUA")
    
    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 3, 1)
    plt.imshow(anh_goc, cmap='gray')
    plt.title('ANH X-RAY GOC', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(anh_clahe, cmap='gray')
    plt.title('SAU CLAHE\n(Tang cuong do tuong phan)', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(anh_morphology, cmap='gray')
    plt.title('SAU MORPHOLOGY\n(Khu nhieu va lam ro)', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(anh_ket_hop, cmap='gray')
    plt.title('KET HOP CLAHE + MORPHOLOGY\n(Ket qua tot nhat)', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(anh_goc, cmap='gray')
    plt.title('ANH GOC', fontsize=12)
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(anh_ket_hop, cmap='gray')
    plt.title('SAU XU LY', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def chay_demo_lam_ro_anh():
    """Chạy demo làm rõ ảnh X-ray"""
    print_f("BAT DAU DEMO LAM RO ANH X-RAY")
    
    thanh_cong, danh_sach_anh = kiem_tra_gdxray_png()
    if not thanh_cong:
        return False
    
    anh_ngau_nhien = chon_anh_ngau_nhien(danh_sach_anh)
    if not anh_ngau_nhien:
        return False
    
    anh_goc = tai_anh_xray(anh_ngau_nhien)
    if anh_goc is None:
        return False
    
    print_f("=== BAT DAU XU LY ANH ===")
    
    anh_clahe, anh_morphology, anh_ket_hop = lam_ro_anh_xray(anh_goc)
    
    print_f("=== HOAN THANH XU LY ===")
    
    hien_thi_ket_qua_ro_net(anh_goc, anh_clahe, anh_morphology, anh_ket_hop)
    
    luu_ket_qua_chat_lu_cao(anh_goc, anh_clahe, anh_ket_hop, os.path.basename(anh_ngau_nhien))
    
    return True

def luu_ket_qua_chat_lu_cao(anh_goc, anh_clahe, anh_ket_hop, ten_file_goc):
    """Lưu kết quả với chất lượng cao"""
    thu_muc_ket_qua = "ket_qua_xu_ly_chat_luong_cao"
    os.makedirs(thu_muc_ket_qua, exist_ok=True)
    
    ten_khong_duoi = os.path.splitext(ten_file_goc)[0]
    
    cv2.imwrite(os.path.join(thu_muc_ket_qua, ten_khong_duoi + "_goc.png"), anh_goc, 
                [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(os.path.join(thu_muc_ket_qua, ten_khong_duoi + "_clahe.png"), anh_clahe,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(os.path.join(thu_muc_ket_qua, ten_khong_duoi + "_ket_hop.png"), anh_ket_hop,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    print_f("DA LUU KET QUA CHAT LUONG CAO VAO: " + thu_muc_ket_qua)

def main():
    """Hàm chính - chạy demo làm rõ ảnh X-ray"""
    print_f("=" * 70)
    print_f("DEMO LAM RO ANH X-RAY - CLAHE & MORPHOLOGY CAI TIEN")
    print_f("=" * 70)
    
    print_f("KIEM TRA THU VIEN")
    try:
        print_f("TAT CA THU VIEN SAN SANG")
    except Exception as e:
        print_f("LOI THU VIEN: " + str(e))
        return
    
    print_f("\n" + "="*50)
    print_f("TU DONG CHON ANH NGAU NHIEN VA XU LY")
    print_f("="*50)
    
    thanh_cong = chay_demo_lam_ro_anh()
    
    if thanh_cong:
        print_f("\n" + "="*50)
        print_f("DEMO HOAN THANH CONG!")
        print_f("ANH X-RAY DA DUOC LAM RO RO NET")
        print_f("KET QUA DA LUU TRONG THU MUC: ket_qua_xu_ly_chat_luong_cao")
        print_f("="*50)
    else:
        print_f("\nDEMO KHONG THUC HIEN DUOC")
        print_f("KIEM TRA LAI DU LIEU VA CHAY LAI")

if __name__ == "__main__":
    main()