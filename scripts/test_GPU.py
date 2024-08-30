import torch

print(torch.__version__)  # 输出 PyTorch 版本
print(torch.cuda.is_available())  # 检查 CUDA 是否可用


print(torch.version.cuda)  # 输出 PyTorch 使用的 CUDA 版本
print(torch.backends.cudnn.version())  # 输出 CuDNN 版本
