import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    # example
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size - (kernel_size - 1) - 1 + 2 * padding) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 16
        # print(f"{convw=} {convh=} {linear_input_size=}")
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.to(device)
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def main():
    model_test = DQN(HEIGHT, WIDTH, 5)
    # model_test.eval()
    x = torch.randn((1, 1, HEIGHT, WIDTH))
    out = model_test(x)
    print(f"Output shape is: {out.shape}")


if __name__ == '__main__':
    main()
