#!/usr/bin/env python3
import os
import gc
import re
import glob
import math
import argparse
import random
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

import chess
import chess.pgn as chess_pgn

from models.CNN import CNN
# from models.CNNLSTM import CNNLSTM
from models.CNNLSTM2 import CNNLSTM2
from models.ResCNN import ResCNN
from utils import *
from board_utils import *

DEFAULT_PAST_POS = 4
DEFAULT_PAST_TIMES = 4
DEFAULT_PREMOVE_THRESH = 0.1
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 256
DEFAULT_LR = 1e-4
DEFAULT_GAMEMODE_ONLY = True
# DEFAULT_GAMEMODE_BASE_MIN = 3*60
# DEFAULT_GAMEMODE_BASE_MAX = 10*60
DEFAULT_GAMEMODE_BASE_MIN = 3*60
DEFAULT_GAMEMODE_BASE_MAX = 5*60
DEFAULT_ELO_MIN = 1000
DEFAULT_ELO_MAX = 9999

MAX_GAMES_PER_SPLIT = 50000
MAX_GAMES_PER_SPLIT_VA = 5000
MAX_GAMES_PER_SPLIT_TEST = 100

def is_GAMEMODE_game(headers, GAMEMODE_base_max=DEFAULT_GAMEMODE_BASE_MAX):
    base, inc = parse_timecontrol(headers.get("TimeControl"))
    return (base is not None) and (base <= float(GAMEMODE_base_max)) and (base >= float(DEFAULT_GAMEMODE_BASE_MIN))

def elo_in_range(headers, elo_min=None, elo_max=None):
    if elo_min is None and elo_max is None:
        return True
    we = to_int_or_none(headers.get("WhiteElo"))
    be = to_int_or_none(headers.get("BlackElo"))
    if we is None or be is None:
        return False
    if (elo_min is not None) and (we < elo_min or be < elo_min):
        return False
    if (elo_max is not None) and (we > elo_max or be > elo_max):
        return False
    return True

def load_inputs_from_game(game, num_past_pos, num_past_times, premove_thresh):
    headers = game.headers

    base, inc = parse_timecontrol(headers.get("TimeControl"))
    we = to_int_or_none(headers.get("WhiteElo")) or 0
    be = to_int_or_none(headers.get("BlackElo")) or 0

    prev_clk = {chess.WHITE: base, chess.BLACK: base}

    last_times = {
        chess.WHITE: deque(maxlen=num_past_times),
        chess.BLACK: deque(maxlen=num_past_times)
    }

    start_board = game.board()
    # pos_hist = deque([board_to_planes(start_board)] * num_past_pos, maxlen=num_past_pos)
    pos_hist = deque([], maxlen=num_past_pos)

    boards, times, labels = [], [], []

    for node in game.mainline():
        if node.parent is None: # root
            continue
        pre_board = node.parent.board()
        mover = pre_board.turn
        opp = not mover

        currentClock = parse_clk_from_comment(node.comment)

        spent = None
        if prev_clk[mover] is None and currentClock is not None:
            prev_clk[mover] = currentClock + (inc or 0.0)
        else:
            if currentClock is not None and prev_clk[mover] is not None:
                spent = time_spent(prev_clk[mover], currentClock, inc)

        if spent is not None:
            last_times[mover].append(spent)
            prev_clk[mover] = currentClock

        have_pos_hist = (len(pos_hist) >= num_past_pos)
        have_time_hist = (len(last_times[mover]) >= num_past_times) and (len(last_times[opp]) >= num_past_times)
        if have_pos_hist and have_time_hist and (spent is not None):
            # BEFORE MOV:
            # last in pos_hist corresponds to pre_board (it does since we append post-board below)
            stack_planes = np.concatenate(list(pos_hist), axis=0)  # (12*num_past_pos,8,8) uint8
            side_to_move = np.full((1, 8, 8), 1 if mover == chess.WHITE else 0, dtype=np.uint8)
            board_input = np.concatenate([stack_planes, side_to_move], axis=0)

            # time calcs BEFORE MOV:
            fullmove = pre_board.fullmove_number
            phase = min(1.0, fullmove/40.0)
            tc = base / DEFAULT_GAMEMODE_BASE_MAX
            prev_self_clk = prev_clk[mover] if prev_clk[mover] is not None else 0.0
            remain_frac = (prev_self_clk / base) if (base and base > 0) else 0.0

            elo_self = float(we if mover == chess.WHITE else be)
            elo_opp  = float(be if mover == chess.WHITE else we)

            extra = np.array([base or 0.0, inc or 0.0, float(fullmove), remain_frac, phase, tc,
                                elo_self, elo_opp], dtype=np.float32)

            self_times = times_vec_from_deque(last_times[mover], num_past_times)
            opp_times = times_vec_from_deque(last_times[opp], num_past_times)
            time_input = np.concatenate([self_times, opp_times, extra], axis=0).astype(np.float32)

            label = 1 if spent <= premove_thresh else 0
            boards.append(board_input) # (N, Cb, 8, 8)
            times.append(time_input) # (N, 2*num_past_times + 8) [self_recent_first..., opp_recent_first..., fluff]
            labels.append(label) # (N,)

        # AFTER MOVE:
        post_board = node.board()
        pos_hist.append(board_to_planes(post_board))

    if len(boards) == 0:
        return None

    return (
        np.stack(boards, axis=0).astype(np.uint8),
        np.stack(times, axis=0).astype(np.float32),
        np.array(labels, dtype=np.uint8),
    )

def load_split(glob_pattern, num_past_pos, num_past_times, premove_thresh,
               max_games=MAX_GAMES_PER_SPLIT, desc="",
               GAMEMODE_only=DEFAULT_GAMEMODE_ONLY,
               GAMEMODE_base_max=DEFAULT_GAMEMODE_BASE_MAX,
               elo_min=DEFAULT_ELO_MIN, elo_max=DEFAULT_ELO_MAX):
    files = sorted(glob.glob(glob_pattern))
    rng = np.random.default_rng(seed=42)
    rng.shuffle(files)
    boards_all, times_all, labels_all = [], [], []
    games_loaded = 0
    games_used = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            while games_used < max_games:
                game = chess_pgn.read_game(f)
                if game is None:
                    break
                try:
                    headers = game.headers

                    if games_loaded % 250 == 1:
                        print(f"  Loaded: {games_used}/{games_loaded}/{len(files)} = {games_used/games_loaded * 100:.2f}%")

                    if GAMEMODE_only and not is_GAMEMODE_game(headers, GAMEMODE_base_max=GAMEMODE_base_max):
                        games_loaded += 1
                        continue

                    if not elo_in_range(headers, elo_min=elo_min, elo_max=elo_max):
                        games_loaded += 1
                        continue

                    ex = load_inputs_from_game(game, num_past_pos, num_past_times, premove_thresh)
                    if ex is None:
                        games_loaded += 1
                        continue

                    b, t, y = ex
                    boards_all.append(b)
                    times_all.append(t)
                    labels_all.append(y)
                    games_loaded += 1
                    games_used += 1
                except Exception:
                    continue
    if len(boards_all) == 0:
        raise RuntimeError(f"No usable games found in '{glob_pattern}'")

    boards = np.concatenate(boards_all, axis=0)
    times  = np.concatenate(times_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    print(f"{desc} loaded: {games_loaded} games, {games_used} eligible games, {len(labels)} move examples")
    return boards, times, labels


class PremoveDataset(Dataset):
    def __init__(self, boards_u8, times_f32, labels_u8, time_mean=None, time_std=None):
        self.boards_u8 = boards_u8 # (N, Cb, 8, 8)
        self.times_f32 = times_f32 # (N, Ct)
        self.labels_u8 = labels_u8 # (N,)
        self.N = boards_u8.shape[0]

        self.time_mean = time_mean
        self.time_std = time_std
        if self.time_mean is None or self.time_std is None:
            self.time_mean = self.times_f32.mean(axis=0)
            self.time_std = self.times_f32.std(axis=0)

        self.time_std = np.where(self.time_std < 1e-6, 1.0, self.time_std)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        b = torch.from_numpy(self.boards_u8[idx].astype(np.float32)) # (Cb,8,8)
        t = (self.times_f32[idx] - self.time_mean) / self.time_std
        t = torch.from_numpy(t.astype(np.float32)) # (Ct,)
        y = torch.tensor(self.labels_u8[idx], dtype=torch.float32)
        return b, t, y


@torch.no_grad()
def r2_from_logits(logits, labels):
    probs = torch.sigmoid(logits)
    y = labels
    y_mean = torch.mean(y)
    sst = torch.sum((y - y_mean) ** 2)
    sse = torch.sum((y - probs) ** 2)
    return float(1.0 - (sse / sst)) if sst > 0 else 0.0

@torch.no_grad()
def prec_rec_fpr(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(torch.int32)
    labels_i = labels.to(torch.int32)

    tp = torch.sum((preds == 1) & (labels_i == 1)).item()
    tn = torch.sum((preds == 0) & (labels_i == 0)).item()
    fp = torch.sum((preds == 1) & (labels_i == 0)).item()
    fn = torch.sum((preds == 0) & (labels_i == 1)).item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    bal_acc   = 1/2 * (tp / (tp + fn) + tn / (tn + fp))
    return precision, recall, fpr, bal_acc



@torch.no_grad()
def update_cm(cm, logits, labels, thr=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= thr)
    labels_i = labels.bool()
    tp = (preds & labels_i).sum().item()
    tn = (~preds & ~labels_i).sum().item()
    fp = (preds & ~labels_i).sum().item()
    fn = (~preds & labels_i).sum().item()
    cm["tp"] += tp
    cm["tn"] += tn
    cm["fp"] += fp
    cm["fn"] += fn


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, amp=False):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    # running_loss = 0.0
    # all_logits = []
    # all_labels = []

    total = 0
    loss_sum = 0.0
    cm = {"tp":0,"tn":0,"fp":0,"fn":0}
    y_sum = 0.0; y2_sum = 0.0  # for R^2
    err_sum = 0.0

    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    for boards, times, labels in loader:

        boards = boards.to(device, non_blocking=True)
        times  = times.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=amp):
            logits = model(boards, times) # -> calls forward() here
            loss = criterion(logits, labels)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

        # running_loss += loss.detach().item() * labels.size(0)
        # all_logits.append(logits.detach().cpu())
        # all_labels.append(labels.detach().cpu())

        loss_sum += loss.item() * labels.size(0)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            y = labels
            y_sum += y.sum().item()
            y2_sum += (y**2).sum().item()
            err_sum += ((y - probs)**2).sum().item()
            update_cm(cm, logits, labels)

        total += labels.size(0)

    # logits_cat = torch.cat(all_logits, dim=0)
    # labels_cat = torch.cat(all_labels, dim=0)
    # avg_loss = running_loss / labels_cat.size(0)

    # precision, recall, fpr, bal_acc = prec_rec_fpr(logits_cat, labels_cat, threshold=0.5)
    # r2 = r2_from_logits(logits_cat, labels_cat)
    # probs = torch.sigmoid(logits_cat)
    # preds = (probs >= 0.5).to(torch.int32)
    # acc = (preds == labels_cat.to(torch.int32)).sum().item() / labels_cat.size(0)

    # return avg_loss, precision, recall, fpr, r2, bal_acc, acc
    avg_loss = loss_sum / total
    y_mean = y_sum / total
    sst = y2_sum - total * (y_mean**2)
    r2 = float(1.0 - (err_sum / sst)) if sst > 0 else 0.0

    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    fpr       = fp / (fp + tn) if fp + tn else 0.0
    bal_acc   = 0.5 * ((tp/(tp+fn) if tp+fn else 0.0) + (tn/(tn+fp) if tn+fp else 0.0))
    acc       = (tp + tn) / max(1, tp+tn+fp+fn)
    return avg_loss, precision, recall, fpr, r2, bal_acc, acc


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = p*targets + (1-p)*(1-targets)
        fl = ( (self.alpha if self.alpha is not None else 1.0) * (1-pt)**self.gamma * ce )
        return fl.mean() if self.reduction=="mean" else fl.sum()

def load_or_cache(cache_path, build_fn):
    if cache_path and os.path.exists(cache_path):
        # print(f"Loading cache {cache_path}")
        data = np.load(cache_path)
        # print(f"Done.")
        return data["boards"], data["times"], data["labels"]
    boards, times, labels = build_fn()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, boards=boards, times=times, labels=labels)
    return boards, times, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_glob", type=str, default="data/training/*.pgn")
    parser.add_argument("--val_glob",   type=str, default="data/validation/*.pgn")
    parser.add_argument("--past_pos",   type=int, default=DEFAULT_PAST_POS)
    parser.add_argument("--past_times", type=int, default=DEFAULT_PAST_TIMES)
    parser.add_argument("--threshold",  type=float, default=DEFAULT_PREMOVE_THRESH)
    parser.add_argument("--epochs",     type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--limit_train_games", type=int, default=MAX_GAMES_PER_SPLIT)
    parser.add_argument("--limit_val_games",   type=int, default=MAX_GAMES_PER_SPLIT_VA)
    parser.add_argument("--limit_test_games",   type=int, default=MAX_GAMES_PER_SPLIT_TEST)
    parser.add_argument("--cache_train", type=str, default="artifacts/train_cache.npz")
    parser.add_argument("--cache_val",   type=str, default="artifacts/val_cache.npz")
    parser.add_argument("--cache_test",   type=str, default="artifacts/test_cache.npz")
    parser.add_argument("--train_max_examples", type=int, default=int(1074577.0/15000.0*47000.0))
    parser.add_argument("--print_model_params", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = True

    boards_tr, times_tr, labels_tr = load_or_cache(
        args.cache_train,
        lambda: load_split(
            args.train_glob, args.past_pos, args.past_times, args.threshold,
            max_games=args.limit_train_games, desc="Train"
        )
    )

    if args.train_max_examples is not None:
        N = len(labels_tr)
        k = N
        if args.train_max_examples is not None:
            k = min(k, int(args.train_max_examples))
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(N, size=k, replace=False)
        boards_tr = boards_tr[idx]
        times_tr  = times_tr[idx]
        labels_tr = labels_tr[idx]
        print(f"Using {len(labels_tr)} / original {len(labels_tr) if args.train_max_examples is None else N} training examples (max={args.train_max_examples})")


    boards_va, times_va, labels_va = load_or_cache(
        args.cache_val,
        lambda: load_split(
            args.val_glob, args.past_pos, args.past_times, args.threshold,
            max_games=args.limit_val_games, desc="Val"
        )
    )

    boards_test, times_test, labels_test = load_or_cache(
        args.cache_test,
        lambda: load_split(
            args.val_glob, args.past_pos, args.past_times, args.threshold,
            max_games=args.limit_test_games, desc="Test"
        )
    )

    time_mean = times_tr.mean(axis=0)
    time_std = times_tr.std(axis=0)
    time_std = np.where(time_std < 1e-6, 1.0, time_std)

    ds_tr = PremoveDataset(boards_tr, times_tr, labels_tr, time_mean=time_mean, time_std=time_std)
    ds_va = PremoveDataset(boards_va, times_va, labels_va, time_mean=time_mean, time_std=time_std)
    ds_test = PremoveDataset(boards_test, times_test, labels_test, time_mean=time_mean, time_std=time_std)

    labels_np = ds_tr.labels_u8.astype(np.int64)
    num_neg = (labels_np == 0).sum()
    num_pos = (labels_np == 1).sum()

    w = np.zeros_like(labels_np, dtype=np.float64)
    w[labels_np == 0] = 0.5 / max(num_neg, 1)
    w[labels_np == 1] = 0.5 / max(num_pos, 1)
    sampler_tr = WeightedRandomSampler(
        weights=torch.from_numpy(w),
        num_samples=len(ds_tr),
        replacement=True
    )

    pin = torch.cuda.is_available()
    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=False, # must be False when using sampler
        sampler=sampler_tr,
        num_workers=0,
        pin_memory=pin,
    )

    # no balancing: va, test
    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    Cb = 12 * args.past_pos + 1
    extras_count = 8
    Ct = 2 * args.past_times + extras_count
    # model = CNN(in_channels_board=Cb, in_dim_times=Ct)
    # model = ResCNN(in_channels_board=Cb, in_dim_times=Ct)
    # model = CNNLSTM(in_channels_board=Cb, in_dim_times=Ct)
    # model = CNNLSTM2(in_channels_board=Cb, in_dim_times=Ct, extras_count=extras_count)
    # model = ResCNN(in_channels_board=Cb, in_dim_times=Ct, ignore_times=True)
    # model = ResCNN(in_channels_board=Cb, in_dim_times=Ct, ignore_history=True)
    model = ResCNN(in_channels_board=Cb, in_dim_times=Ct, ignore_times=True, ignore_history=True)
    # model = ResCNN(in_channels_board=Cb, in_dim_times=Ct, ignore_board=True)
    # model = ResCNN(in_channels_board=Cb, in_dim_times=Ct, ignore_board=True, ignore_times=True)

    if args.print_model_params:
        model.print_model_param_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pos = labels_tr.sum()
    pos_va = labels_va.sum()
    pos_test = labels_test.sum()
    neg = len(labels_tr) - pos
    pos_weight = torch.tensor([ (neg / max(pos, 1)) ], dtype=torch.float32, device=device)

    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = FocalLoss(gamma=2.0, alpha=0.75)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)


    print(f"Device: {device}")
    print(f"Train examples: {len(ds_tr)}; Val examples: {len(ds_va)}; "
          f"Positives (train): {int(pos)}/{len(labels_tr)}; "
          f"Positives (valid): {int(pos_va)}/{len(labels_va)}; "
          f"Positives (test): {int(pos_test)}/{len(labels_test)}")

    baselineAcc = (1 - int(pos_va)/len(labels_va)) * 100

    # Train

    best_bal_acc = float("-inf")
    no_improve_epochs = 0
    patience = 5

    collected_objects = gc.collect()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_prec, tr_rec, tr_fpr, tr_r2, tr_bal_acc, tr_acc = run_epoch(
            model, dl_tr, device, criterion, optimizer, scheduler=None, amp=True
        )
        va_loss, va_prec, va_rec, va_fpr, va_r2, va_bal_acc, va_acc = run_epoch(
            model, dl_va, device, criterion, optimizer=None, scheduler=None, amp=True
        )

        test_loss, test_prec, test_rec, test_fpr, test_r2, test_bal_acc, test_acc = run_epoch(
            model, dl_test, device, criterion, optimizer=None, scheduler=None, amp=True
        )

        collected_objects = gc.collect()

        # print(f"garbage collected {collected_objects} objects")


        print(f"[Epoch {epoch:02d}/{args.epochs}] "
              f"Train: loss={tr_loss:.4f}, bal_acc={tr_bal_acc*100:.1f}%, acc={tr_acc*100:.1f}%, "
              f"Prec={tr_prec*100:.1f}%, Rec={tr_rec*100:.1f}%, FPR={tr_fpr*100:.1f}, R2={tr_r2:.4f} "
              f"| Val: loss={va_loss:.4f}, bal_acc={va_bal_acc*100:.1f}%, acc={va_acc*100:.1f}% (baseline: {baselineAcc:.1f}%), "
              f"Prec={va_prec*100:.1f}%, Rec={va_rec*100:.1f}%, FPR={va_fpr*100:.1f}%, R2={va_r2:.4f}"
              f"| Test: loss={test_loss:.4f}, bal_acc={test_bal_acc*100:.1f}%, acc={test_acc*100:.1f}% (baseline: {baselineAcc:.1f}%), "
              f"Prec={test_prec*100:.1f}%, Rec={test_rec*100:.1f}%, FPR={test_fpr*100:.1f}%, R2={test_r2:.4f}")
        # print(f"[Epoch {epoch:02d}/{args.epochs}] "
        #       f"Train: bal_acc={tr_bal_acc*100:.1f}%, "
        #       f"Prec={tr_prec*100:.1f}%, Rec={tr_rec*100:.1f}%, FPR={tr_fpr*100:.1f}, R2={tr_r2:.4f} "
        #       f"| Val: bal_acc={va_bal_acc*100:.1f}% "
        #       f"Prec={va_prec*100:.1f}%, Rec={va_rec*100:.1f}%, FPR={va_fpr*100:.1f}%, R2={va_r2:.4f}"
        #       f"| Test: bal_acc={test_bal_acc*100:.1f}%, "
        #       f"Prec={test_prec*100:.1f}%, Rec={test_rec*100:.1f}%, FPR={test_fpr*100:.1f}%, R2={test_r2:.4f}")

        # if epoch == 7:
        #     break

        if va_bal_acc > best_bal_acc:
            best_bal_acc = va_bal_acc
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"early stopping epoch {epoch}")
                break



    os.makedirs("artifacts", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        # "config": {
        #     "past_pos": args.past_pos, "past_times": args.past_times,
        #     "time_mean": time_mean, "time_std": time_std, "Cb": Cb, "Ct": Ct
        # }
    }, "artifacts/premove_CNN.pt")
    print("saved to artifacts/premove_CNN.pt")

if __name__ == "__main__":
    main()
