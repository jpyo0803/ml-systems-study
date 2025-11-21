import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleMLP()
model.eval()

x = torch.randn(1, 4)
traced = torch.jit.trace(model, x)
traced.save("./model_repository/simple_mlp/1/model.pt")



