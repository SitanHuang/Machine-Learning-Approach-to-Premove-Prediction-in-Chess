#!/usr/bin/env python3
import os, random, pathlib

random.seed(1387)

def read_games(pgn_path):
    games, cur = [], []
    with open(pgn_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("[Event ") and cur:
                games.append("".join(cur).strip() + "\n")
                cur = [line]
            else:
                cur.append(line)
        if cur:
            g = "".join(cur).strip()
            if g:
                games.append(g + "\n")
    return [g for g in (g.strip()+"\n" for g in games) if g.strip()]

def split_and_write(games, out_root, prefix):
    games = games[:]
    random.shuffle(games)
    n = len(games)

    n_test = int(max(0.01 * n, 1000))
    n_test = min(n_test, n)
    test, remaining = games[:n_test], games[n_test:]

    n_train = int(len(remaining) * 0.9)
    train, val = remaining[:n_train], remaining[n_train:]

    train_dir = pathlib.Path(out_root, "training")
    val_dir   = pathlib.Path(out_root, "validation")
    test_dir  = pathlib.Path(out_root, "test")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for i, g in enumerate(train, 1):
        (train_dir / f"{prefix}_{i:05d}.pgn").write_text(g, encoding="utf-8")
    for i, g in enumerate(val, 1):
        (val_dir / f"{prefix}_{i:05d}.pgn").write_text(g, encoding="utf-8")
    for i, g in enumerate(test, 1):
        (test_dir / f"{prefix}_{i:05d}.pgn").write_text(g, encoding="utf-8")

def main():
    root = "."
    all_pgns = [f for f in os.listdir(root) if f.lower().endswith(".pgn")]
    if not all_pgns:
        print("No PGN files found")
        return

    print("Available PGN files:")
    for idx, f in enumerate(all_pgns, 1):
        print(f"{idx}: {f}")

    selected = input("Which PGN files? (or 'all'): ")
    if selected.strip().lower() == "all":
        chosen = all_pgns
    else:
        nums = [int(x) for x in selected.split(",") if x.strip().isdigit()]
        chosen = [all_pgns[i-1] for i in nums if 1 <= i <= len(all_pgns)]

    if not chosen:
        print("No files selected.")
        return

    for fname in chosen:
        prefix = pathlib.Path(fname).stem
        games = read_games(fname)
        split_and_write(games, root, prefix)
        print(f"{fname}: {len(games)} games -> training/, validation/, test/")

if __name__ == "__main__":
    main()
