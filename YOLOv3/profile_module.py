import torch
import torch.nn as nn
import time
from thop import profile
from collections import OrderedDict
from nets.yolo import YoloBody
import math


# ==============================================================================
# 步骤 2: 主评测逻辑 (和上次完全一致)
# ==============================================================================
if __name__ == "__main__":
    # 1. 配置参数
    input_shape = [512, 512]  # 输入尺寸 512x512
    num_classes = 80          
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    
    # 2. 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. 创建完整模型，并提取待测试的模块
    full_model = YoloBody(anchors_mask, num_classes)
    
    # 核心：提取出你想要测试的模块 self.TMM
    # module_to_test = full_model.compress_model
    module_to_test = full_model.TMM
    # module_to_test = full_model
    # 参数量 (Params)  : 63.8117 M
    # 计算量 (GFLOPs)  : 88.5571 G
    # 平均时延 (Latency) : 62.1457 ms
    



    module_to_test.to(device)
    module_to_test.eval()

    # 4. 创建符合模块输入的伪数据 (dummy input)
    # dummy_input1 = torch.randn(1, 16, input_shape[0], input_shape[1]).to(device)
    # dummy_input2 = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)

    dummy_input1 = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    dummy_input2 = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)

    print(f"评测设备: {device}")
    print(f"输入尺寸: {dummy_input1.shape}")
    print(f"输入尺寸: {dummy_input2.shape}")
    print("\n" + "="*50)
    print("      开始评测模块: self.TMM (Adaptive_Module)")
    print("="*50 + "\n")

    # 5. 计算 GFLOPs 和参数量
    with torch.no_grad():
        flops, params = profile(module_to_test, inputs=(dummy_input1,dummy_input2,), verbose=False)
        

    gflops = flops / 1e9
    m_params = params / 1e6
    
    print(f"模块参数量 (Params): {m_params:.4f} M")
    print(f"理论计算量 (GFLOPs): {gflops:.4f} G")
    
    # 6. 测量推理时延 (Latency)
    print("\n--- 正在测量推理时延 (ms) ---")
    num_warmup = 20
    num_runs = 100
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = module_to_test(dummy_input1,dummy_input2)

        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        total_time = 0
        for _ in range(num_runs):
            start_time = time.time()
            _ = module_to_test(dummy_input1,dummy_input2)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time)
            
    average_latency_ms = (total_time / num_runs) * 1000
    
    print("\n--- 评测结果汇总 ---")
    print(f"模块           : self.TMM (Adaptive_Module)")
    print(f"参数量 (Params)  : {m_params:.4f} M")
    print(f"计算量 (GFLOPs)  : {gflops:.4f} G")
    print(f"平均时延 (Latency) : {average_latency_ms:.4f} ms")
    print(f"参考FPS          : {1000 / average_latency_ms:.2f}")