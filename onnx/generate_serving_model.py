import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, X):
        return self.net(X)

class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout),
        )

    def forward(self, X):
        return self.net(X)

class MNISTClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(1, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            Flatten(),
            FCBlock(64*7*7, 128),
            nn.Linear(128, num_classes),
        )

    def forward(self, X):
        return self.net(X)

if __name__ == '__main__':
    model = MNISTClassifier(num_classes=10)
    model.load_state_dict(torch.load('./mnist_classifier.pth'))
    model.eval()

    X = torch.randn(1, 1, 28, 28) # warm up

    # ONNX 서빙용 모델 저장

    torch.onnx.export(
        model,
        X,
        'mnist_classifier.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print("Finish export: mnist_classifier.onnx")
    

