import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available. GPU is ready to use.")
else:
    print("CUDA is not available. GPU will not be used.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建网络模型
class Feng(torch.nn.Module):
    def __init__(self):
        super(Feng, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 64),
            torch.nn.Linear(64, 10),

        )

    def forward(self, x):
        x = self.model(x)
        return x


feng = Feng()
# 先判断一下是否有GPU进行训练，但是每一个使用GPU前，都需要进行判断。
if torch.cuda.is_available():
    # 调用GPU进行训练
    feng = feng.cuda()

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
print("done")
