"""
train_100k.py
Targets 100,000 pipes. Loads existing champion as base.
Prints every single generation clearly.
Uses half your cores so laptop stays cool.
"""
import os, sys, json, time, math, random, copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

WIDTH=380; HEIGHT=580; GROUND_HEIGHT=70
GRAVITY=0.35; JUMP_FORCE=-6.5; PIPE_SPEED=2.4
PIPE_GAP=145; PIPE_WIDTH=55; PIPE_INTERVAL=110
TARGET=100_000; FLOOR=HEIGHT-GROUND_HEIGHT
POPULATION=60

def sigmoid(x):
    if x>40: return 1.0
    if x<-40: return 0.0
    return 1/(1+math.exp(-x))

def brain_from_js(d):
    return {"wih":d["weightsIH"],"who":d["weightsHO"],"bh":d["biasH"],"bo":d["biasO"]}

def brain_to_js(b):
    return {"weightsIH":b["wih"],"weightsHO":b["who"],"biasH":b["bh"],"biasO":b["bo"]}

def mutate(b, rate=0.12, strength=0.25):
    c=copy.deepcopy(b)
    for i in range(6):
        for j in range(5):
            if random.random()<rate: c["wih"][i][j]+=random.uniform(-strength,strength)
        if random.random()<rate: c["bh"][i]+=random.uniform(-strength,strength)
    for j in range(6):
        if random.random()<rate: c["who"][0][j]+=random.uniform(-strength,strength)
    if random.random()<rate: c["bo"][0]+=random.uniform(-strength,strength)
    return c

def run_trial(args):
    brain, seed = args
    if seed: random.seed(seed)
    bird_y=float(FLOOR//2); vel=0.0; bird_x=65.0; score=0; frames=0
    min_h=50; max_h=FLOOR-PIPE_GAP-min_h
    N=20; px=[0.0]*N; pt=[0.0]*N; pb=[0.0]*N; pp=[True]*N
    head=tail=active=0
    def spawn(x):
        nonlocal head,active
        th=random.randint(min_h,max_h)
        px[head]=float(x); pt[head]=float(th)
        pb[head]=float(th+PIPE_GAP); pp[head]=False
        head=(head+1)%N; active+=1
    spawn(WIDTH)
    while score<TARGET:
        frames+=1
        if frames%PIPE_INTERVAL==0: spawn(WIDTH)
        cx=1e9; ct=0.35; cb=0.65; idx=tail
        for _ in range(active):
            d=(px[idx]+PIPE_WIDTH)-bird_x
            if 0.0<d<cx: cx=d; ct=pt[idx]/HEIGHT; cb=pb[idx]/HEIGHT
            idx=(idx+1)%N
        w=brain["wih"]; bh=brain["bh"]; v=brain["who"][0]
        i0=bird_y/HEIGHT; i1=(vel+10)/20; i2=cx/WIDTH; i3=ct; i4=cb
        h0=sigmoid(i0*w[0][0]+i1*w[0][1]+i2*w[0][2]+i3*w[0][3]+i4*w[0][4]+bh[0])
        h1=sigmoid(i0*w[1][0]+i1*w[1][1]+i2*w[1][2]+i3*w[1][3]+i4*w[1][4]+bh[1])
        h2=sigmoid(i0*w[2][0]+i1*w[2][1]+i2*w[2][2]+i3*w[2][3]+i4*w[2][4]+bh[2])
        h3=sigmoid(i0*w[3][0]+i1*w[3][1]+i2*w[3][2]+i3*w[3][3]+i4*w[3][4]+bh[3])
        h4=sigmoid(i0*w[4][0]+i1*w[4][1]+i2*w[4][2]+i3*w[4][3]+i4*w[4][4]+bh[4])
        h5=sigmoid(i0*w[5][0]+i1*w[5][1]+i2*w[5][2]+i3*w[5][3]+i4*w[5][4]+bh[5])
        if sigmoid(h0*v[0]+h1*v[1]+h2*v[2]+h3*v[3]+h4*v[4]+h5*v[5]+brain["bo"][0])>0.5:
            vel=JUMP_FORCE
        vel+=GRAVITY
        if vel>9: vel=9.0
        bird_y+=vel
        if bird_y+13>=FLOOR: break
        if bird_y-13<=0: bird_y=13; vel=0.0
        alive=True; evict=0; idx=tail
        for _ in range(active):
            px[idx]-=PIPE_SPEED
            if not pp[idx] and px[idx]+PIPE_WIDTH<bird_x: pp[idx]=True; score+=1
            if bird_x+13>px[idx] and bird_x-13<px[idx]+PIPE_WIDTH:
                if bird_y-13<pt[idx] or bird_y+13>pb[idx]: alive=False; break
            if px[idx]+PIPE_WIDTH<-10: evict+=1
            idx=(idx+1)%N
        if not alive: break
        if evict: tail=(tail+evict)%N; active-=evict
    return score

def save(brain, score, gen):
    with open("best_bird.json","w") as f:
        json.dump({"record_score":score,"generation":gen,"brain":brain_to_js(brain)},f,indent=2)

def main():
    # Use half cores — keep laptop cool
    num_cores = max(2, (os.cpu_count() or 4) // 2)

    print("=" * 65)
    print(f" 🎯 TARGET: 100,000 PIPES | Starting from saved champion")
    print(f" 🖥️  Using {num_cores} / {os.cpu_count()} cores (laptop-friendly)")
    print(f" 🐦  Population: {POPULATION} birds per generation")
    print("=" * 65)

    # Load champion
    champion = None
    best_score = 0
    if os.path.exists("best_bird.json"):
        with open("best_bird.json") as f:
            d = json.load(f)
        champion   = brain_from_js(d["brain"])
        best_score = d.get("record_score", 0)
        print(f"✅ Loaded champion — previous record: {best_score:,} pipes\n")
    else:
        print("⚠️  No saved brain found, starting fresh\n")
        champion = {"wih":[[random.uniform(-1,1) for _ in range(5)] for _ in range(6)],
                    "who":[[random.uniform(-1,1) for _ in range(6)]],
                    "bh":[random.uniform(-1,1) for _ in range(6)],
                    "bo":[random.uniform(-1,1)]}

    if best_score >= TARGET:
        print(f"🏆 Already at {best_score:,} pipes — already past 100k!")
        return

    print(f"{'Gen':>5} | {'This Gen':>10} | {'Record':>10} | {'Time':>7} | {'Top 3 Scores':<30} | Note")
    print("-" * 85)
    sys.stdout.flush()

    generation = 0
    start_time = time.time()

    try:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            while best_score < TARGET:
                generation += 1

                # Build population
                candidates = [champion]
                for _ in range(POPULATION - 1):
                    r = random.choice([0.06, 0.10, 0.18, 0.28])
                    s = random.choice([0.12, 0.22, 0.40, 0.65])
                    candidates.append(mutate(champion, rate=r, strength=s))

                seeds   = [random.randint(1, 9_999_999) for _ in candidates]
                results = list(executor.map(run_trial, zip(candidates, seeds)))

                ranked            = sorted(zip(results, candidates), key=lambda x: x[0], reverse=True)
                gen_best, gen_brain = ranked[0]
                top3              = [s for s,_ in ranked[:3]]
                top3_str          = "  ".join(f"{s:>8,}" for s in top3)
                elapsed           = time.time() - start_time
                note              = ""

                if gen_best > best_score:
                    best_score = gen_best
                    champion   = gen_brain
                    save(champion, best_score, generation)
                    note = "🌟 NEW RECORD!"
                    if best_score >= TARGET:
                        note = "🏆 100K ACHIEVED!"

                survivors = sum(1 for s in results if s > 0)
                pct       = min(100.0, best_score / TARGET * 100)

                print(f"{generation:>5} | {gen_best:>10,} | {best_score:>10,} | "
                      f"{elapsed:>6.1f}s | {top3_str} | {note}")

                # Progress bar every 10 gens
                if generation % 10 == 0:
                    bar_n = int(pct/2)
                    bar   = "█"*bar_n + "░"*(50-bar_n)
                    avg   = sum(results)/len(results)
                    print(f"      ╰─ [{bar}] {pct:.1f}% | avg:{avg:.0f} | alive:{survivors}/{POPULATION}")
                    print("-" * 85)

                sys.stdout.flush()

                if best_score >= TARGET:
                    break

    except KeyboardInterrupt:
        print("\n⛔ Stopped by user.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    if best_score >= TARGET:
        print(f" 🏆💥 100,000 PIPES CLEARED! Done in {elapsed:.1f}s!")
    else:
        print(f" 📊 Best reached: {best_score:,} pipes | {elapsed:.1f}s")
    print("=" * 65)
    save(champion, best_score, generation)
    print(f"\n💾 Saved to best_bird.json!")
    print("👉 Go to http://localhost:8000 → 📥 Load from Disk")

if __name__ == "__main__":
    mp.freeze_support()
    main()
