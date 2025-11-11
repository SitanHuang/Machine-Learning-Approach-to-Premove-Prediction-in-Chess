import torch
import torch.nn as nn

class CNNLSTM2(nn.Module):
    def __init__(
        self,
        in_channels_board: int, # = 12*T + 1
        in_dim_times: int, # = 2*T + extras
        extras_count: int = 8,
        pieces_channel_count: int = 12,
        board_cnn_channels: int = 256,
        board_out_dim: int = 128,
        board_lstm_hidden: int = 128,
        time_step_out: int = 32,
        time_lstm_hidden: int = 32,
        extras_out: int = 32,
        head_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.extras_count = extras_count

        assert (in_channels_board - 1) % pieces_channel_count == 0
        T_pos = (in_channels_board - 1) // pieces_channel_count
        assert (in_dim_times - extras_count) % 2 == 0 and (in_dim_times - extras_count) >= 2
        T_time = (in_dim_times - extras_count) // 2

        assert T_pos == T_time
        self.T = T_pos
        self.pieces_channel_count = pieces_channel_count

        self.boardCNN = nn.Sequential(
            nn.Conv2d(pieces_channel_count, board_cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(board_cnn_channels),
            nn.Conv2d(board_cnn_channels, board_cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(board_cnn_channels),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), # (B*T, board_cnn_channels)
        )
        self.boardHead = nn.Sequential(
            nn.Linear(board_cnn_channels, board_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.boardLSTM = nn.LSTM(
            input_size=board_out_dim,
            hidden_size=board_lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # each step is [self_t, opp_t]
        self.timeStepMLP = nn.Sequential(
            nn.Linear(2, time_step_out),
            nn.ReLU(inplace=True),
        )
        self.timeLSTM = nn.LSTM(
            input_size=time_step_out,
            hidden_size=time_lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.extrasMLP = nn.Sequential(
            nn.Linear(extras_count, extras_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        concat_dim = board_lstm_hidden + time_lstm_hidden + extras_out
        self.head = nn.Sequential(
            nn.Linear(concat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, 1),
        )

    def split_inputs(
        self, boards: torch.Tensor, times: torch.Tensor
    ):
        extras_count = self.extras_count
        # !!!!!! TIME VEC MUST BE RECENT-FIRST ORDER (B, 2*T + 8)
        B, Cb, H, W = boards.shape
        assert H == 8 and W == 8

        x = boards[:, : Cb - 1] # (B, 12*T, 8, 8)
        # (B, T, 12, 8, 8)
        T = self.T
        board_seq = x.reshape(B, T, self.pieces_channel_count, H, W)

        Ct = times.shape[1]
        T_from_times = (Ct - extras_count) // 2
        assert T_from_times == T
        self_times = times[:, :T]
        opp_times = times[:, T:2*T]
        extras = times[:, 2*T:2*T + extras_count] # (B, extras_count)

        # remember we did "last_times[mover].append(spent)" ???
        # well the "recent = list(reversed(recent))" is performed at times_vec_from_deque so we should be gucci here

        # newest->oldest to oldest->newest
        self_times = torch.flip(self_times, dims=[1]) # (B, T)
        opp_times = torch.flip(opp_times, dims=[1]) # (B, T)
        time_seq = torch.stack([self_times, opp_times], dim=2) # (B, T, 2)

        return board_seq, time_seq, extras

    def forward(self, boards: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        B = boards.size(0)
        board_seq, time_seq, extras = self.split_inputs(boards, times)

        extras_count = self.extras_count

        BT = B * self.T
        b = board_seq.reshape(BT, self.pieces_channel_count, 8, 8) # (B*T, 12, 8, 8)
        b = self.boardCNN(b) # (B*T, C)
        b = self.boardHead(b) # (B*T, board_emb_dim)
        b = b.reshape(B, self.T, -1) # (B, T, board_emb_dim)
        _, (hB, _) = self.boardLSTM(b) # hB: (1, B, board_lstm_hidden)
        hB = hB[-1] # (B, board_lstm_hidden)

        t = self.timeStepMLP(time_seq) # (B, T, time_step_emb)
        _, (hT, _) = self.timeLSTM(t) # (1, B, time_lstm_hidden)
        hT = hT[-1] # (B, time_lstm_hidden)

        e = self.extrasMLP(extras) # (B, extras_emb)

        out = self.head(torch.cat([hB, hT, e], dim=1)).squeeze(1) # (B,)
        return out

    def print_model_param_size(self):
        total_params = 0
        print("Model Parameter Summary:")
        print("-" * 40)
        for name, p in self.named_parameters():
            if p.requires_grad:
                n = p.numel()
                total_params += n
                print(f"{name:40s} {list(p.shape)} -> {n:,} params")
        print("-" * 40)
        print(f"Total trainable parameters: {total_params:,}")
        print(f"Total size (in MB): {total_params * 4 / (1024 ** 2):.2f} MB")
