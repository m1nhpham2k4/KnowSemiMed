import matplotlib.pyplot as plt
import pandas as pd

# Cấu hình hiển thị font chữ và độ phân giải cao
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

# --- KHỞI TẠO DỮ LIỆU CHO 3 BẢNG ---

# Bảng 1
data1 = {
    "Phương pháp": ["Mean Teacher (2 U-Net)", "BCP (Mean Teacher + 2 U-Net)", "SemiSAM", "KnowSAM (Bình thường)", "KnowSAM (MedSAM Chỉ 1 U-Net)", "U-Net 100% Label"],
    "5% Dice": ["0.8357", "0.7996", "0.8007", "0.8714", "0.8802", "0.9098"],
    "5% IoU": ["0.7537", "0.7079", "0.7083", "0.7878", "0.8012", "0.8449"],
    "5% HD95": ["15.58", "22.33", "25.35", "4.84", "4.64", "5.98"],
    "10% Dice": ["0.8643", "0.8495", "0.7850", "0.8782", "0.8886", "-"],
    "30% Dice": ["0.9028", "0.8943", "0.8456", "0.8938", "0.8991", "-"]
}

# Bảng 2
data2 = {
    "Phương pháp": ["Mean Teacher (2 U-Net)", "BCP (Mean Teacher + 2 U-Net)", "SemiSAM", "KnowSAM (Bình thường)", "KnowSAM (MedSAM)", "U-Net 100% Label"],
    "5% Dice": ["0.2462", "0.2543", "0.3910", "0.4710", "0.3913", "0.7698"],
    "5% IoU": ["0.1875", "0.1748", "0.3226", "0.3734", "0.3076", "0.6995"],
    "5% HD95": ["85.67", "91.43", "57.07", "5.73", "6.68", "24.83"],
    "10% Dice": ["0.4599", "0.3708", "0.4761", "0.5824", "0.6215", "-"],
    "30% Dice": ["0.7105", "0.5253", "0.6349", "0.7238", "0.6895", "-"]
}

# Bảng 3
data3 = {
    "Method": ["Mean Teacher (2 V-Net)", "BCP (Mean Teacher + 2 U-Net)", "SemiSAM (MedSAM)", "KnowSAM (Bình thường)", "KnowSAM (SAM2)", "U-Net 100% Label"],
    "5% Dice": ["0.7679", "0.7598", "0.7541", "0.7544", "0.7860", "0.8886"],
    "5% IoU": ["0.6590", "0.6594", "0.6502", "0.6381", "0.6733", "0.8148"],
    "5% HD95": ["13.73", "13.22", "18.82", "4.01", "3.97", "7.24"],
    "10% Dice": ["0.8414", "0.7627", "0.8106", "0.8228", "0.8063", "-"],
    "30% Dice": ["0.6690", "0.8708", "0.8252", "0.8378", "0.8456", "-"]
}

dfs = [pd.DataFrame(data1), pd.DataFrame(data2), pd.DataFrame(data3)]
titles = ["BẢNG 1: LEFT ATRIUM DATASET", "BẢNG 2: MULTI-ORGAN DATASET", "BẢNG 3: PROMISE12 DATASET"]

# --- KHỞI TẠO ĐỒ HỌA CHỮ NÉT CAO ---
fig, axes = plt.subplots(1, 3, figsize=(26, 10))
fig.patch.set_facecolor('#ffffff')

for i, ax in enumerate(axes):
    ax.axis('off')
    df = dfs[i]
    
    # Tạo bảng biểu thực tế trong không gian vector
    table = ax.table(
        cellText=df.values, 
        colLabels=df.columns, 
        cellLoc='center', 
        loc='center'
    )
    
    # Thiết kế UI chuyên nghiệp (Kiểu McKinsey/Academic)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5) # Giúp các hàng dãn cách thoáng, dễ đọc
    
    # Định dạng màu sắc và kẻ đường viền sắc nét
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#d1d5db') # Viền xám mảnh hiện đại
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#0f172a') # Thanh tiêu đề xanh Slate đậm
        else:
            # Highlight các dòng kết quả xuất sắc của KnowSAM
            if "KnowSAM" in df.iloc[row-1, 0]:
                cell.set_facecolor('#f0fdf4') # Nền xanh lá cực nhạt cực sang
                cell.set_text_props(weight='semibold')
            elif "100%" in df.iloc[row-1, 0]:
                cell.set_facecolor('#f8fafc') # Dòng tham chiếu màu xám nhạt
            else:
                cell.set_facecolor('#ffffff')
                
    ax.set_title(titles[i], fontsize=15, weight='bold', pad=30, color='#1e293b')

plt.tight_layout()

# Xuất file ảnh với độ phân giải siêu nét (DPI=400) chống vỡ hình hoàn toàn
plt.savefig("combined_metrics_high_res.png", bbox_inches='tight', dpi=400, facecolor=fig.get_facecolor())
print("Đã tạo xong file ảnh siêu nét 'combined_metrics_high_res.png'!")