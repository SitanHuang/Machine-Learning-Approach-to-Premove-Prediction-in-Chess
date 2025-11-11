import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.actv(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.actv(x + y)

class ResCNN(nn.Module):
    def __init__(self, in_channels_board: int, in_dim_times: int,
                 blocks: int = 32, num_blocks: int = 8,
                 ignore_times: bool = False,
                 ignore_history: bool = False,
                 ignore_board: bool = False,
                 board_planes_per_pos: int = 12,
                 extras_count: int = 8):

        super().__init__()
        self.ignore_times = ignore_times
        self.ignore_history = ignore_history
        self.ignore_board = ignore_board
        self.extras_count = extras_count
        self.time_dim = max(0, in_dim_times - extras_count)

        self.curr_board_channels = board_planes_per_pos + 1 # 1 stm
        stem_in_channels = self.curr_board_channels if ignore_history else in_channels_board

        board_out_dim = 128

        if self.ignore_board:
            self.board_head = nn.Identity()
            board_out_dim = 0
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(stem_in_channels, blocks, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(blocks),
                nn.ReLU(inplace=True),
            )
            self.tower = nn.Sequential(*[ResidualBlock(blocks) for _ in range(num_blocks)])
            self.gap = nn.AdaptiveMaxPool2d((1, 1))
            self.board_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(blocks, board_out_dim),
                nn.ReLU(inplace=True),
            )

        extras_out_dim = 32
        self.extras_head = nn.Sequential(
            nn.Linear(self.extras_count, extras_out_dim),
            nn.ReLU(inplace=True),
        )

        if not self.ignore_times and self.time_dim > 0:
            self.time_head = nn.Sequential(
                nn.Linear(self.time_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
            )
            time_out_dim = 16
        else:
            self.time_head = nn.Identity()
            time_out_dim = 0

        self.head = nn.Sequential(
            nn.Linear(board_out_dim + extras_out_dim + time_out_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, boards, times):
        if self.ignore_history:
            boards = boards[:, -self.curr_board_channels:, :, :]

        if not self.ignore_board:
            x = self.stem(boards)
            x = self.tower(x)
            x = self.gap(x)
            xb = self.board_head(x)

        if times is None:
            raise ValueError("times vec none")
        if self.extras_count > 0:
            extras = times[:, -self.extras_count:]
        else:
            extras = times.new_zeros((times.size(0), 0))
        xe = self.extras_head(extras)

        if not self.ignore_times and self.time_dim > 0:
            time_hist = times[:, :self.time_dim]
            xt = self.time_head(time_hist)
            if self.ignore_board:
                feats = torch.cat([xe, xt], dim=1)
            else:
                feats = torch.cat([xb, xe, xt], dim=1)
        else:
            if self.ignore_board:
                feats = torch.cat([xe], dim=1)
            else:
                feats = torch.cat([xb, xe], dim=1)

        out = self.head(feats)
        return out.squeeze(1)

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
        print(
            f"Total size (in MB): {total_params * 4 / (1024 ** 2):.2f} MB"
            f"  ignore_board={self.ignore_board}"
            f"  ignore_times={self.ignore_times}"
            f"  ignore_history={self.ignore_history}"
            f"  extras_count={self.extras_count}"
            f"  time_dim={self.time_dim}"
            f"  curr_board_channels={self.curr_board_channels}"
        )