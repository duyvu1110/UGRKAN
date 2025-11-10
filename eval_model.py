from ugrkan import UGRKAN
import torch
import torch.nn as nn
from thop import profile
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UGRKAN(1).to(device)

INPUT_H = 256
INPUT_W = 256

# 2. Tạo một input giả (dummy input)
# Kích thước: (batch_size, in_channels, height, width)
# in_channels của bạn là 1 (từ UGRKAN(1, 256))
try:
    dummy_input = torch.randn(1, 1, INPUT_H, INPUT_W).to(device)
    print(f"Đang phân tích model UGRKAN với đầu vào: (1, 1, {INPUT_H}, {INPUT_W})")

    # 3. Tính toán FLOPs và Params
    # 'verbose=False' để tắt log chi tiết từng layer
    flops, params = profile(model, inputs=(dummy_input, ), verbose=False)

    # 4. In kết quả
    print("\n--- 📊 Kết quả tính toán ---")
    print(f"  Input size:    (1, 1, {INPUT_H}, {INPUT_W})")
    print(f"  Parameters:    {params / 1e6:.3f} M")
    print(f"  GFLOPs:        {flops / 1e9:.3f} G")

    # Kiểm tra với con số 10.9M params của bạn
    if not (10.8 < (params / 1e6) < 11.0):
        print("\n[CẢNH BÁO]: Số params (Triệu) tính được không khớp với 10.9M bạn đã nêu.")
        print("Điều này có nghĩa là 'thop' có thể đã bỏ qua layer GRKAN tùy chỉnh.")

except Exception as e:
    print(f"\n[LỖI] Không thể thực hiện profile:")
    print(f"  {e}")
    print("\nLưu ý: Thư viện 'thop' rất có thể KHÔNG hỗ trợ các layer tùy chỉnh (custom operations) như GRKAN.")
    print("Nếu điều này xảy ra, con số GFLOPs sẽ không chính xác (hoặc bằng 0).")