import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels_board, in_dim_times):
        super().__init__()
        self.boardCNN = nn.Sequential(
            nn.Conv2d(in_channels_board, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(), # (256,)
        )
        self.boardHead = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.timeMLP = nn.Sequential(
            nn.Linear(in_dim_times, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            # nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, boards, times):
        xb = self.boardHead(self.boardCNN(boards))
        xt = self.timeMLP(times)
        logit = self.head(torch.cat([xb, xt], dim=1)).squeeze(1)
        return logit

    def print_model_param_size(self):
        total_params = 0
        print("Model Parameter Summary:")
        print("-" * 40)
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_size = param.numel()
                total_params += param_size
                print(f"{name:40s} {list(param.shape)} -> {param_size:,} params")
        print("-" * 40)
        print(f"Total trainable parameters: {total_params:,}")
        print(f"Total size (in MB): {total_params * 4 / (1024 ** 2):.2f} MB")
