# Reinforcement Learning for Traffic Signal Control (CTDE Double DQN)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![Eclipse SUMO](https://img.shields.io/badge/Eclipse%20SUMO-Traffic%20Simulation-brightgreen.svg)](https://eclipse.dev/sumo/)

An intelligent traffic light control system built using a **Centralised Training, Decentralised Execution (CTDE)** architecture. This project leverages a **Dueling Double Deep Q-Network (DDQN)** to optimize traffic flow across a dynamically generated 4x4 urban grid in the Eclipse SUMO microscopic traffic simulator.

## 🌟 Key Features

* **CTDE Architecture:** A single globally-trained "Master Brain" optimizes a shared replay buffer and neural network policy, while local decentralized agents execute actions based on their local geometric state and neighboring intersection data.
* **Dueling Double DQN:** Separates state-value calculation from action-advantage calculation to better identify inherently congested states, utilizing Double Q-learning to prevent overestimation biases.
* **Custom Composite Reward Function:** Balances overall system delay with a quadratic starvation penalty (ensuring fairness for low-traffic lanes) and a wasted-green penalty.
* **Deterministic Safety Gatekeeper:** A hard-coded rule engine that enforces minimum/maximum green limits and orchestrates yellow-light transitions automatically, bypassing the neural network computationally when a switch is mathematically forced.
* **Dynamic Grid Generation:** Automated Python scripts to mathematically generate SUMO XML networks, routes, and varying traffic flow patterns (Vertical Rush, Horizontal Rush, Chaos).

## 📂 Project Structure

```text
├── env/
│   ├── setup.py        # Generates the 4x4 SUMO grid, routes, and XML config
│   ├── simple.sumocfg      # SUMO configuration file for the RL agent
│   ├── simple_actuated.sumocfg      # SUMO configuration file for an actuated baseline
│   └── simple_fixed.sumocfg      # SUMO configuration file for a fixed baseline
├── src/
│   ├── ctde_agent.py       # Core DQN Agent, Replay Buffer, and Neural Net topologies
│   ├── main.py             # Training loop, CTDE orchestration, and plotting logic
│   ├── evaluation.py       # Automated evaluation against the 2 baselines
│   └── test_agent.py       # Automated unit tests
├── outputs/
│   ├── evaluation_reports/ # Auto-generated reports from evaluation runs
│   └── learning_curves/    # Auto-generated Matplotlib training graphs
├── models/                 # Saved PyTorch global policy weights (.pth)
├── README.md
└── requirements.txt
```

## ⚙️ Installation & Requirements

1. **Eclipse SUMO:** You must have SUMO installed and the `SUMO_HOME` environment variable configured.
   * [Download SUMO](https://eclipse.dev/sumo/)
2. **Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: The project attempts to use the faster C++ `libsumo` API if available, but will automatically fallback to the standard Python `traci` module.*

## 🚀 Usage

### 1. Generate the Environment
Before training, generate the geometric 4x4 grid, E2 loop detectors, and traffic flow routes:
```bash
cd env
python setup.py
```

### 2. Train the RL Agents
Run the main training loop. The script orchestrates the SUMO simulation, synchronizes the Double DQN target networks, and decays the exploration rate (Epsilon).
```bash
cd src
python main.py
```
*(To watch the agents learn visually, uncomment `import traci` inside `main.py` to trigger the SUMO-GUI instead of the headless backend).*

### 3. Evaluate the Trained Model
Bechmark the learned policy against the fixed-time and actuated baselines using the evaluation script.
```bash
cd src
python evaluation.py
```

## 🧠 System Architecture Overview

### State Space (29 Dimensions)
Each agent normalizes real-time loop detector data into a strict `[0.0, 1.0]` tensor. Observations include:
* Current phase identity (One-hot encoded) and normalized time spent in the current phase.
* Lane queue lengths.
* Approaching vehicles and downstream physical blockages.
* Upstream traffic data filtered dynamically from neighboring nodes (N, S, E, W).
* Real-time tracking of "Starved" lanes to penalize excessive waiting times.

### Action Space (4 Discrete Actions)
The agent selects between 4 distinct green-light phases (Standard N/S, Protected Left N/S, Standard E/W, Protected Left E/W). Yellow phases are completely abstracted away from the AI and handled safely by the Gatekeeper.

### The Safety Gatekeeper
Located inside the agent's `step()` function, this logic ensures:
1. **Min Green:** Bypasses action selection if the light just turned green.
2. **Max Green (Anti-Starvation):** Mathematically forces a phase shift, bypassing the PyTorch forward pass to save compute time, while accurately recording the forced transition into the Replay Buffer.
3. **Yellow Transitions:** Overrides target phase requests to safely transition through intermediate yellow states.
