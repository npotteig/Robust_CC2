# Adaptive Cyber-Defense for Evolving Attacker Capabilities using Reinforcement Learning and Behavior Trees

## Setup
```bash
uv sync
```

Set `N_JOBS` and `N_THREADS` in `blue_ppo_train.py`, `red_ppo_train.py`, and `evaluation.py` according to how many vCPUS you would like to allocate for increased parallelism.

## Train and evaluate attacker (red) agents
```bash
uv run bash full_red_inductive_cardiff.sh
uv run bash full_red_inductive_keep.sh
```

```bash
uv run bash full_red_evaluation_cardiff.sh
uv run bash full_red_evaluation_keep.sh
```

## Train and evaluate defender (blue) agents
Update `full_blue_transfer.sh` and `full_blue_evaluation.sh` with red and blue checkpoints you would like to use.

```bash
uv run bash full_blue_transfer.sh
```

```bash
uv run bash full_blue_evaluation.sh
```

