from collections import deque, defaultdict
import re
import chess
import numpy as np

CLK_RE = re.compile(r'%clk\s+([0-9:\.]+)')

def parse_timecontrol(tc_str):
    if not tc_str or tc_str.strip() in ("-", ""):
        return (None, 0.0)
    s = tc_str.strip()
    base, inc = None, 0.0
    if "+" in s:
        a, b = s.split("+", 1)
        base = int(a)
        b = b.strip()
        m = re.match(r'^(\d+(?:\.\d+)?)', b)
        inc = float(m.group(1)) if m else 0.0
    elif "|" in s:
        a, b = s.split("|", 1)
        base = int(a)
        m = re.match(r'^(\d+(?:\.\d+)?)', b)
        inc = float(m.group(1)) if m else 0.0
    else:
        base = int(s)
        inc = 0.0
    return (float(base) if base is not None else None, float(inc))

def parse_clk_from_comment(comment):
    if not comment:
        return None
    m = CLK_RE.search(comment)
    if not m:
        return None
    ts = m.group(1)
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            hh = int(parts[0]); mm = int(parts[1]); ss = float(parts[2])
            return 3600*hh + 60*mm + ss
        elif len(parts) == 2:
            mm = int(parts[0]); ss = float(parts[1])
            return 60*mm + ss
        else:
            return float(parts[0])
    except Exception:
        return None

def time_spent(prev_clk, curr_clk, inc):
    if prev_clk is None or curr_clk is None:
        return None
    spent = prev_clk - curr_clk + (inc or 0.0)
    if spent < 0:
        spent = 0.0
    return float(spent)

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def board_to_planes(board: chess.Board) -> np.ndarray:
    planes = []
    for color in (chess.WHITE, chess.BLACK):
        for piece in PIECE_ORDER:
            plane = np.zeros((8, 8), dtype=np.uint8)
            for sq in board.pieces(piece, color):
                r = 7 - chess.square_rank(sq)  # rank 8 -> row 0
                c = chess.square_file(sq) # file a -> col 0
                plane[r, c] = 1
            planes.append(plane)


    # [White P,N,B,R,Q,K, Black P,N,B,R,Q,K], (12, 8, 8)
    return np.stack(planes, axis=0)  # (12,8,8)

def times_vec_from_deque(dq: deque, L: int) -> np.ndarray:
    recent = list(dq)[-L:]
    recent = list(reversed(recent)) # most recent first
    v = np.zeros(L, dtype=np.float32)
    v[:len(recent)] = np.array(recent, dtype=np.float32)
    return v
