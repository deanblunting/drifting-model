# Quantum State Tomography via Drifting Models

Implementation of [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) (Deng et al., 2026) applied to quantum state tomography.

## Overview

This project uses the Drifting Model framework to learn quantum measurement statistics via one-step generative modeling. A neural network generator maps noise to measurement outcome distributions, trained using a drifting field that attracts generated samples toward real measurement data and repels them from other generated samples. At equilibrium (V=0), the generated distribution matches the true quantum statistics.

**Phase 1:** Single-qubit tomography (proof of concept)  
**Phase 2:** Multi-qubit tomography (2-8 qubits, entanglement, incomplete measurements)

## Setup

```bash
cd /Users/dean/Developer/papers/drifting-model
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

```bash
# Phase 1: Single qubit
python experiments/phase1_single_qubit.py

# Phase 2: Multi-qubit experiments
python experiments/phase2_multi_qubit.py

# Phase 2: Incomplete tomography (fewer bases than needed)
python experiments/phase2_incomplete.py

# Run all tests
python -m pytest tests/ -v
```

## Project Structure

```
quantum_drifting/
  states.py         — Quantum state simulation (1-8 qubits)
  drifting.py        — Drifting field computation (Algorithm 2)
  generator.py       — Generator network (conditioned on measurement basis)
  trainer.py         — Training loop with logging
  tomography.py      — State reconstruction from learned distributions
  utils.py           — Visualization and helpers
experiments/
  phase1_single_qubit.py
  phase2_multi_qubit.py
  phase2_incomplete.py
tests/
  test_states.py
  test_drifting.py
```

## Hardware

Designed for M3 Pro with 18GB RAM. All experiments use MPS (Metal) acceleration when available, with automatic CPU fallback.
