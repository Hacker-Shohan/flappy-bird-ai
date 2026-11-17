# Flappy Bird AI

A Flappy Bird clone controlled by a lightweight 43-parameter neural network trained with neuroevolution using elitism + mutation (a (1+λ) evolution strategy). The bundled champion model achieved a record run of 100,000 consecutive pipes during training and achieves an 85.5% survival rate (9,241 mean pipes) across 200 fresh benchmark seeds.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Play%20in%20Browser-brightgreen)](https://hacker-shohan.github.io/flappy-bird-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Live Demo

Play or watch the trained model directly in your browser:  
👉 **[https://hacker-shohan.github.io/flappy-bird-ai/](https://hacker-shohan.github.io/flappy-bird-ai/)**

- **Zero dependencies**: Pure vanilla JavaScript, HTML5 Canvas, and modern CSS. No build step or package manager required.
- **Embedded model**: The pre-trained champion weights are embedded directly into `index.html`.
- **Live Brain Synapses Visualizer**: Dynamic rendering of the network's 5 inputs, 6 hidden neurons, and flap output with real-time weight coloring and node activations.
- **Simulation Controls**: 1x to 100x speed slider and pause toggle.
- **Difficulty Stress-Testing**: Live slider with Easy, Normal, Hard, and Extreme presets to test the model against tighter pipe gaps (down to 115px) and faster spawn frequencies (down to 80 frames).

---

## How It Works

### 1. Neural Network Architecture

A compact feedforward Multi-Layer Perceptron (MLP) controls flap decisions every frame:

- **Input Layer (5 nodes)**:
  1. `Bird Y`: Normalized bird height (`y / canvas.height`)
  2. `Velocity`: Normalized vertical velocity (`(vel + 10) / 20`)
  3. `Pipe Distance`: Normalized horizontal distance to next pipe (`(pipe.x - bird.x) / canvas.width`)
  4. `Gap Top`: Normalized vertical coordinate of upper pipe boundary (`gapTop / canvas.height`)
  5. `Gap Bottom`: Normalized vertical coordinate of lower pipe boundary (`gapBottom / canvas.height`)
- **Hidden Layer (6 neurons)**: Weighted sum + bias, activated via standard Sigmoid.
- **Output Layer (1 neuron)**: Sigmoid probability output.
  - Output $> 0.5 \implies$ **Flap** (impulse = $-6.5$)
  - Output $\le 0.5 \implies$ **Glide** (accelerated by gravity $= +0.35$)

$$\text{Output} = \sigma\left( \sum_{j=1}^6 (\text{Hidden}_j \times W_{HO,j}) + b_O \right)$$

Total parameter count: **43 weights and biases** (30 input-hidden + 6 hidden-output + 6 hidden biases + 1 output bias).

---

### 2. Neuroevolution: (1+λ) Evolution Strategy

Code analysis of the trainer (`train.py`) reveals key architectural details:

1. **Evolution Strategy, Not Genetic Algorithm**: The population of 60 candidate networks is formed by taking 1 preserved champion (elitism) and generating $\lambda = 59$ offspring strictly through Gaussian mutations of the champion's weights. **There is no crossover or multi-parent recombination**, making this a $(1+\lambda)$ evolution strategy rather than a traditional genetic algorithm.
2. **RNG Seeding & Fitness Landscape**: During headless training, candidates evaluate on independently randomized seeds. While this introduces stochastic evaluation noise (some candidates encounter easier pipe sequences than others), elitism ensures surviving weights continually refine a robust, general-purpose gap-seeking policy.
3. **Fitness Metric**: Fitness equals the number of pipes cleared before ground or pipe collision.

---

## Evaluation & Empirical Benchmark

While the champion model scored **100,000 continuous pipes** on a single training seed run, real-world generalization requires multi-seed statistical testing. We evaluated `best_bird.json` headlessly using `evaluate.py` across **200 fresh random seeds**, capping each trial at **10,000 pipes**.

### Simulation Parameters
- **Pipe Gap**: 145 px (vertical opening between upper and lower pipes)
- **Pipe Spacing**: 110 frames (~264 px horizontal distance at 2.4 px/frame)
- **Pipe Speed**: 2.4 px/frame
- **Gap Height Range**: Uniform random between $y = 50$ px and $y = 315$ px
- **Bird Physics**: Gravity = 0.35, Jump Impulse = -6.5, Max Velocity = 9.0

### Benchmark Results (200 Seeds, 10,000-Pipe Cap)

| Metric | Result |
|---|---|
| **Seeds Tested** | 200 |
| **Survival Rate (reached 10k cap)** | **85.5%** (171 / 200 seeds) |
| **Mean Pipes Cleared** | **9,241.6 pipes** |
| **Median Pipes Cleared** | **10,000.0 pipes** |
| **Minimum Pipes Cleared** | **130 pipes** |
| **Maximum Pipes Cleared** | **10,000 pipes** |

### Honest Failure Analysis
Across 200 arbitrary seeds, **29 seeds failed** prior to the 10,000-pipe mark (e.g. Seed 38 failed at pipe 152; Seed 67 at pipe 779; Seed 45 at pipe 916; Seed 148 at pipe 1,343). 

Why does this happen?
- **No Temporal Memory**: The network has only 43 static weights with zero recurrent states (RNN/LSTM) and no history buffer of past frames.
- **Momentum Trapping**: When procedural generation randomly places consecutive pipes at opposite vertical extremes (e.g. extreme bottom gap immediately followed by extreme top gap), the bird's downward velocity can exceed the recovery window of fixed flap impulses.

Run the evaluation suite locally:
```bash
python evaluate.py -n 200 -m 10000
```

---

## Running Locally

### Option 1: Standalone Web Interface
Double-click `index.html` or open it in any web browser. No server, node, or npm required.

### Option 2: Headless Python Training
Run multi-process headless training:

```bash
python train.py --target 100000
```

To plot training metrics recorded in `training_log.csv`:
```bash
python plot_log.py
```

---

## Project Structure

```
.
├── index.html        # Self-contained showcase web app & live visualizer
├── train.py          # Multi-process (1+λ) neuroevolution trainer
├── evaluate.py       # Headless multi-seed statistical benchmark
├── plot_log.py       # Training curve visualization (matplotlib)
├── best_bird.json    # Exported champion weights (100k pipe record)
├── assets/           # Media and demonstration assets
├── README.md         # Documentation & benchmark analysis
├── LICENSE           # MIT License
└── .gitignore
```

---

## Limitations

- **Single Champion Architecture**: Offspring are derived solely from 1 champion with varying mutation strengths. There is no multi-lineage gene pool or genetic crossover.
- **Fixed, Small Topology**: The network topology is fixed at 5-6-1 (43 parameters). It cannot dynamically discover new features, connections, or hidden layers.
- **Physics Sensitivity**: The model was trained specifically on fixed simulation physics (Gravity = 0.35, Jump = -6.5, Speed = 2.4, Gap = 145px). Aggressive difficulty presets or differing framerates impact performance.
- **Reactive, Markovian Policy**: Decisions depend strictly on instantaneous state inputs with zero memory of previous trajectory frames.

---

## Future Ideas

- **Genetic Crossover**: Implement two-parent crossover operators (arithmetic or uniform recombination) across diverse elite sub-populations.
- **NEAT Benchmark**: Implement NeuroEvolution of Augmenting Topologies (NEAT) to evolve network topology alongside connection weights.
- **Deep RL Comparisons**: Benchmark sample efficiency and collision resilience against Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO).
- **Domain Randomization**: Train on randomized physics parameters (varying gravity, flap force, and speed) to produce an omni-capable generalist.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
