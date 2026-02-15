# Quantum State Tomography via Drifting Models

Implementation of [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) (Deng et al., 2026) applied to quantum state tomography.

## Overview

This project uses the Drifting Model framework to learn quantum measurement statistics via one-step generative modeling. A neural network generator maps noise to measurement outcome distributions, trained using a drifting field that attracts generated samples toward real measurement data and repels them from other generated samples. At equilibrium (V=0), the generated distribution matches the true quantum statistics.

The approach works well for 1-3 qubit systems (fidelity >0.84) but does not scale beyond this due to the exponential growth of the output dimension causing kernel distances to concentrate.

**Phase 1:** Single-qubit tomography (proof of concept)
**Phase 2:** Multi-qubit tomography (2-3 qubits, entanglement, incomplete measurements)
**Phase 3:** 4-qubit tomography (demonstrates scaling limitations)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

```bash
# Phase 1: Single qubit
python experiments/phase1_single_qubit.py

# Phase 2: Multi-qubit (Bell, GHZ, W states, 2-4 qubits)
python experiments/phase2_multi_qubit.py

# Phase 2: Incomplete tomography (fewer bases than needed)
python experiments/phase2_incomplete.py

# Phase 3: 4-qubit GHZ with all 81 bases
python experiments/phase3_4qubit.py

# Run all tests
python -m pytest tests/ -v
```

## Project Structure

```
quantum_drifting/
  states.py      - Quantum state simulation (1-8 qubits)
  drifting.py    - Drifting field computation (Algorithm 2)
  generator.py   - SimpleGenerator MLP (conditioned on measurement basis)
  trainer.py     - Training loop with logging
  tomography.py  - State reconstruction from learned distributions
  utils.py       - Visualization and helpers
experiments/
  phase1_single_qubit.py
  phase2_multi_qubit.py
  phase2_incomplete.py
  phase3_4qubit.py
tests/
  test_states.py
  test_drifting.py
```

## Hardware

Designed for M3 Pro. All experiments use MPS (Metal) acceleration when available, with automatic CPU fallback.
