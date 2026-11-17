# Flappy Bird AI

A Flappy Bird clone controlled by a neural network trained with neuroevolution using elitism + mutation (a (1+λ) evolution strategy). The bundled model clears over 100,000 consecutive pipes without crashing.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Play%20in%20Browser-brightgreen)](https://hacker-shohan.github.io/flappy-bird-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Live Demo

Play or watch the trained model in your browser:  
👉 **[https://hacker-shohan.github.io/flappy-bird-ai/](https://hacker-shohan.github.io/flappy-bird-ai/)**

- **Zero dependencies**: Pure vanilla JavaScript, HTML5 Canvas, and CSS. No build step or package manager required.
- **Pre-trained model**: The 100,000-pipe weights are embedded directly into `index.html`.
- **Interactive UI**: Includes a real-time neural network visualizer, bird tracking roster, speed controls, and a toggle between pre-trained demo mode and live evolution.

---

## How It Works

### 1. Neural Network Architecture

A feedforward neural network (Multi-Layer Perceptron) controls the bird's flap decisions every frame:

- **Input Layer (5 nodes)**:
  1. `Bird Y` (normalized bird height)
  2. `Velocity` (normalized vertical velocity)
  3. `Pipe Distance` (horizontal distance to upcoming pipe)
  4. `Gap Top` (vertical position of the upper pipe boundary)
  5. `Gap Bottom` (vertical position of the lower pipe boundary)
- **Hidden Layer (6 neurons)**: Computes weighted sums with biases, squashed by the Sigmoid activation function.
- **Output Layer (1 neuron)**: Sigmoid output representing flap probability.
  - Output $> 0.5 \implies$ **Flap**
  - Output $\le 0.5 \implies$ **Glide**

$$\text{Output} = \sigma\left( \sum (\text{Hidden} \times W_{\text{output}}) + b_{\text{output}} \right)$$

Total parameters: **43 weights and biases**.

---

### 2. Neuroevolution: (1+λ) Evolution Strategy

Instead of backpropagation or crossover-based genetic algorithms, the model learns via a **(1+λ) evolution strategy** (elitism + mutation):

1. **Population**: A generation of candidate birds (default: 60) is evaluated in parallel.
2. **Selection & Elitism**: The single best-performing champion is preserved unchanged into the next generation.
3. **Mutation**: The remaining $\lambda = 59$ offspring are generated solely by mutating the champion's weights and biases with varying mutation rates and perturbation strengths (no crossover is used).
4. **Fitness Metric**: Evaluated by the number of pipes cleared across seeded procedural obstacles.

Over successive generations, the network converges on an optimal flap frequency and gap-targeting trajectory.

---

## Running Locally

### Option 1: Standalone Web Interface
Open `index.html` in any web browser. No local web server required.

### Option 2: Headless Python Training
To run headless multi-core training:

```bash
# Clone the repository
git clone https://github.com/Hacker-Shohan/flappy-bird-ai.git
cd flappy-bird-ai

# Run headless trainer
python train.py
```

The script evaluates generations in parallel using Python's `multiprocessing` module, logs progress to `training_log.csv`, and saves records to `best_bird.json`.

---

## Evaluation & Benchmark

The 100,000-pipe milestone was achieved during continuous headless training on a procedural seed stream. To evaluate generalizability, the model was tested headlessly across **200 fresh random seeds**, with each trial capped at **10,000 pipes**:

### Simulation Parameters
- **Pipe Gap**: 145 px (vertical opening between upper and lower pipes)
- **Pipe Spacing**: 110 frames (~264 px horizontal distance at 2.4 px/frame)
- **Pipe Speed**: 2.4 px/frame
- **Gap Height Range**: Procedural uniform distribution between `y = 50` and `y = 315` px (`canvas.height - GROUND_HEIGHT - PIPE_GAP - minHeight`)
- **Bird Physics**: Gravity = 0.35, Jump Impulse = -6.5, Max Fall Velocity = 9.0

### Benchmark Results (200 Seeds, 10k Cap)
- **Survival Rate**: **85.5%** (171 / 200 seeds reached the 10,000-pipe cap without crashing)
- **Mean Pipes Cleared**: **9,241.6 pipes**
- **Median Pipes Cleared**: **10,000.0 pipes**
- **Min Pipes Cleared**: **130 pipes**
- **Max Pipes Cleared**: **10,000 pipes**

### Failure Analysis
Across the 200 seeds, **29 seeds failed** prior to the 10,000-pipe mark (e.g., Seed 38 failed at pipe 152; Seed 67 at pipe 779; Seed 45 at pipe 916; Seed 148 at pipe 1,343). Because this minimal Multi-Layer Perceptron has only 43 parameters with no temporal memory (RNN/LSTM) or frame history, rare consecutive extreme high-to-low pipe gap placements can catch the bird in an unrecoverable vertical velocity state.

Run the evaluation suite locally:
```bash
python evaluate.py -n 200 -m 10000
```

---

## Project Structure

```
.
├── index.html        # Self-contained web game, visualizer, and dashboard
├── train.py          # Multi-process headless trainer
├── evaluate.py       # Headless multi-seed evaluation benchmark
├── plot_log.py       # Training curve visualization (matplotlib)
├── best_bird.json    # Exported champion weights (100k pipe score)
├── README.md         # Documentation
├── LICENSE           # MIT License
└── .gitignore
```

---

## License

This project is open-source and available under the [MIT License](LICENSE).
