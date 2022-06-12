import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):

    def __init__(self, h, w, outputs=8):
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
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.to(device)
        x = x.float()
        x = x.view(-1, 1, HEIGHT, WIDTH)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class LargerDQN(nn.Module):

    def __init__(self, h, w, outputs=8):
        super(LargerDQN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(8192, 4096)
        self.dropout1 = nn.Dropout()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout()

        self.fc3 = nn.Linear(4096, outputs)

    def forward(self, x):
        x = x.to(device)
        x = x.float()
        x = x.view(-1, 1, HEIGHT, WIDTH)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


def main():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_test = DQN(HEIGHT, WIDTH, 8)
    model_test.to(device)
    x = torch.randn((1, 1, HEIGHT, WIDTH))
    out = model_test(x)
    print(f"Output shape is: {out.shape}")

    large_model = LargerDQN(HEIGHT, WIDTH, 8)
    large_model.to(device)
    x2 = torch.randn((1, 1, HEIGHT, WIDTH))
    out = large_model(x2)


if __name__ == '__main__':
    main()
