"""
evaluate.py
Robust evaluation script for Flappy Bird Neuroevolution Model.
Evaluates best_bird.json headlessly across multiple random seeds
under the exact same physics and procedural pipe generation as the game.
"""

import os
import sys
import json
import math
import random
import argparse
import statistics
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Simulation Constants (matching index.html & train.py)
WIDTH = 380
HEIGHT = 580
GROUND_HEIGHT = 70
GRAVITY = 0.35
JUMP_FORCE = -6.5
PIPE_SPEED = 2.4
PIPE_GAP = 145
PIPE_WIDTH = 55
PIPE_INTERVAL = 110
FLOOR = HEIGHT - GROUND_HEIGHT


def sigmoid(x):
    if x > 40:
        return 1.0
    if x < -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def brain_from_js(d):
    return {
        "wih": d["weightsIH"],
        "who": d["weightsHO"],
        "bh": d["biasH"],
        "bo": d["biasO"]
    }


def evaluate_seed(args):
    brain, seed, max_pipes = args
    if seed is not None:
        random.seed(seed)

    bird_y = float(FLOOR // 2)
    vel = 0.0
    bird_x = 65.0
    score = 0
    frames = 0

    min_h = 50
    max_h = FLOOR - PIPE_GAP - min_h

    N = 20
    px = [0.0] * N
    pt = [0.0] * N
    pb = [0.0] * N
    pp = [True] * N
    head = tail = active = 0

    def spawn(x):
        nonlocal head, active
        th = random.randint(min_h, max_h)
        px[head] = float(x)
        pt[head] = float(th)
        pb[head] = float(th + PIPE_GAP)
        pp[head] = False
        head = (head + 1) % N
        active += 1

    spawn(WIDTH)

    while score < max_pipes:
        frames += 1
        if frames % PIPE_INTERVAL == 0:
            spawn(WIDTH)

        cx = 1e9
        ct = 0.35
        cb = 0.65
        idx = tail
        for _ in range(active):
            d = (px[idx] + PIPE_WIDTH) - bird_x
            if 0.0 < d < cx:
                cx = d
                ct = pt[idx] / HEIGHT
                cb = pb[idx] / HEIGHT
            idx = (idx + 1) % N

        w = brain["wih"]
        bh = brain["bh"]
        v = brain["who"][0]

        i0 = bird_y / HEIGHT
        i1 = (vel + 10) / 20
        i2 = cx / WIDTH
        i3 = ct
        i4 = cb

        h0 = sigmoid(i0 * w[0][0] + i1 * w[0][1] + i2 * w[0][2] + i3 * w[0][3] + i4 * w[0][4] + bh[0])
        h1 = sigmoid(i0 * w[1][0] + i1 * w[1][1] + i2 * w[1][2] + i3 * w[1][3] + i4 * w[1][4] + bh[1])
        h2 = sigmoid(i0 * w[2][0] + i1 * w[2][1] + i2 * w[2][2] + i3 * w[2][3] + i4 * w[2][4] + bh[2])
        h3 = sigmoid(i0 * w[3][0] + i1 * w[3][1] + i2 * w[3][2] + i3 * w[3][3] + i4 * w[3][4] + bh[3])
        h4 = sigmoid(i0 * w[4][0] + i1 * w[4][1] + i2 * w[4][2] + i3 * w[4][3] + i4 * w[4][4] + bh[4])
        h5 = sigmoid(i0 * w[5][0] + i1 * w[5][1] + i2 * w[5][2] + i3 * w[5][3] + i4 * w[5][4] + bh[5])

        out = sigmoid(h0 * v[0] + h1 * v[1] + h2 * v[2] + h3 * v[3] + h4 * v[4] + h5 * v[5] + brain["bo"][0])
        if out > 0.5:
            vel = JUMP_FORCE

        vel += GRAVITY
        if vel > 9:
            vel = 9.0
        bird_y += vel

        # Collision with floor
        if bird_y + 13 >= FLOOR:
            break

        # Ceiling clamp
        if bird_y - 13 <= 0:
            bird_y = 13
            vel = 0.0

        alive = True
        evict = 0
        idx = tail
        for _ in range(active):
            px[idx] -= PIPE_SPEED
            if not pp[idx] and px[idx] + PIPE_WIDTH < bird_x:
                pp[idx] = True
                score += 1

            if bird_x + 13 > px[idx] and bird_x - 13 < px[idx] + PIPE_WIDTH:
                if bird_y - 13 < pt[idx] or bird_y + 13 > pb[idx]:
                    alive = False
                    break

            if px[idx] + PIPE_WIDTH < -10:
                evict += 1

            idx = (idx + 1) % N

        if not alive:
            break

        if evict:
            tail = (tail + evict) % N
            active -= evict

    return seed, score


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Flappy Bird AI model across random seeds.")
    parser.add_argument("--model", type=str, default="best_bird.json", help="Path to best_bird.json")
    parser.add_argument("-n", "--seeds", type=int, default=200, help="Number of random seeds to evaluate (default: 200)")
    parser.add_argument("-m", "--max-pipes", type=int, default=10_000, help="Max pipes cleared per trial (default: 10,000)")
    parser.add_argument("--seed-start", type=int, default=1, help="Starting integer for seed sequence (default: 1)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Parallel worker processes")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)

    with open(args.model, "r") as f:
        data = json.load(f)

    brain = brain_from_js(data["brain"])
    claimed_score = data.get("record_score", "Unknown")

    print("=" * 65)
    print(" 🧪 Flappy Bird AI Model Evaluation Benchmark")
    print(f" Model: {args.model} (Claimed record: {claimed_score:,} pipes)")
    print(f" Evaluation: {args.seeds} seeds | Cap: {args.max_pipes:,} pipes/seed")
    print(f" Workers: {args.workers} processes")
    print(f" Pipe Gap: {PIPE_GAP}px | Pipe Interval: {PIPE_INTERVAL} frames | Speed: {PIPE_SPEED}")
    print("=" * 65)

    seed_list = [args.seed_start + i for i in range(args.seeds)]
    tasks = [(brain, s, args.max_pipes) for s in seed_list]

    scores = []
    failed_seeds = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_seed, t): t[1] for t in tasks}
        for future in as_completed(futures):
            seed, score = future.result()
            scores.append(score)
            completed += 1
            if score < args.max_pipes:
                failed_seeds.append((seed, score))
            if completed % 25 == 0 or completed == args.seeds:
                pct = (completed / args.seeds) * 100
                print(f"  Progress: {completed}/{args.seeds} seeds ({pct:.1f}%) evaluated...", flush=True)

    survived = sum(1 for s in scores if s >= args.max_pipes)
    survival_rate = (survived / len(scores)) * 100.0
    mean_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    min_score = min(scores)
    max_score = max(scores)

    print("\n📊 Benchmark Results:", flush=True)
    print(f"  • Total seeds tested:    {len(scores):,}", flush=True)
    print(f"  • Survived (>= cap):     {survived:,} / {len(scores):,} ({survival_rate:.1f}%)", flush=True)
    print(f"  • Mean pipes cleared:    {mean_score:,.1f}", flush=True)
    print(f"  • Median pipes cleared:  {median_score:,.1f}", flush=True)
    print(f"  • Min pipes cleared:     {min_score:,}", flush=True)
    print(f"  • Max pipes cleared:     {max_score:,}", flush=True)

    if failed_seeds:
        print(f"\n⚠️  Failed Seeds ({len(failed_seeds)}):", flush=True)
        for s, sc in sorted(failed_seeds)[:20]:
            print(f"    Seed {s}: failed at pipe {sc:,}", flush=True)
        if len(failed_seeds) > 20:
            print(f"    ... and {len(failed_seeds) - 20} more.", flush=True)
    else:
        print("\n✅ Zero failures: 100% of seeds reached the max pipe cap without crashing!", flush=True)

    print("=" * 65, flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
