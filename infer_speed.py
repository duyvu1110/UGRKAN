from ugrkan import UGRKAN
import torch
import torch.nn as nn
import time
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_fp32 = UGRKAN(
    num_classes=1,
).to(device)
model_fp32.eval() # Set to evaluation mode
INPUT_H = 256
INPUT_W = 256
# Create the dummy input for the FP32 model
dummy_input_fp32 = torch.randn(1, 3, INPUT_H, INPUT_W).to(device)

print(f"FP32 model defined. Dtype: {next(model_fp32.parameters()).dtype}")

# --- 3. Define the FP16 (Quantized) Model ---
model_fp16 = copy.deepcopy(model_fp32)

# BÂY GIỜ CHUYỂN ĐỔI BẢN SAO SANG FP16
model_fp16 = model_fp16.half()
model_fp16.eval()
dummy_input_fp16 = dummy_input_fp32.half()

print(f"FP16 model defined. Dtype: {next(model_fp16.parameters()).dtype}")

def benchmark(model, input_tensor, model_name):
    """
    Runs a benchmark for a given model and input.
    """
    print(f"\n--- Benchmarking {model_name} ---")
    
    # 1. Warm-up runs
    print("Warming up...")
    for _ in range(20):
        with torch.no_grad():
            _ = model(input_tensor)
            
    # Wait for all warm-up runs to finish
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
    
    # 2. Timed runs
    N_RUNS = 100
    print(f"Running {N_RUNS} timed iterations...")
    
    start_time = time.time()
    for _ in range(N_RUNS):
        with torch.no_grad():
            _ = model(input_tensor)
            
    # 3. CRITICAL: Wait for all GPU operations to complete
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
    
    end_time = time.time()
    
    # 4. Calculate results
    total_time = end_time - start_time
    avg_time_ms = (total_time / N_RUNS) * 1000 # Average time in milliseconds
    fps = N_RUNS / total_time
    
    print(f"Avg. Inference Time: {avg_time_ms:.3f} ms")
    print(f"Inference Speed (FPS): {fps:.2f}")
    
    return fps, avg_time_ms


try:
    fp32_fps, fp32_ms = benchmark(model_fp32, dummy_input_fp32, "FP32 (Normal)")
    fp16_fps, fp16_ms = benchmark(model_fp16, dummy_input_fp16, "FP16 (Quantized)")
    
    print("\n--- 📊 Final Results ---")
    print(f"FP32 (Normal):   {fp32_ms:.3f} ms/frame  ({fp32_fps:.2f} FPS)")
    print(f"FP16 (Quantized): {fp16_ms:.3f} ms/frame  ({fp16_fps:.2f} FPS)")
    
    speedup = fp16_fps / fp32_fps
    print(f"\nSpeedup (FP16 vs FP32): {speedup:.2f}x")

except Exception as e:
    print(f"\nAn error occurred during benchmarking: {e}")