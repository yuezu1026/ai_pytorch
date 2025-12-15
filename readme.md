### 关于pytorch的学习和研究
1. 如何让代码能执行带cuda上

import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available. GPU is ready to use.")
else:
    print("CUDA is not available. GPU will not be used.")

# 创建一个张量
x = torch.randn(3, 3)

# 将张量移动到GPU的第一个设备（如果有多个GPU，可以通过指定不同的设备ID来选择）
x = x.to('cuda')  # 或者使用x.cuda()

# 如果你有多个GPU，可以选择特定的GPU
x = x.to('cuda:0')  # 选择GPU 0


print(x)