# Topological Liquidity Trap: Monte Carlo Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the simulation code accompanying the paper:

**"Topological Phase Transitions in Over-the-Counter Markets: A Structural View of Liquidity Collapse"**

## Overview

This code implements Monte Carlo simulations to validate the theoretical predictions about liquidity collapse in OTC markets. The key finding is that when the fraction of active agents falls below a critical percolation threshold, systemic liquidity vanishes regardless of asset prices—a phenomenon we term the **"Topological Liquidity Trap."**

## Key Concepts

- **Market Spacetime**: A directed random graph model with stochastic node participation
- **Max-Plus Spectral Radius**: The maximum average weight of a cycle, representing asymptotic growth rate
- **Thresholded Systemic Liquidity**: A metric that distinguishes local circulation from systemic circulation
- **Phase Transition**: Sharp transition at critical threshold p_c = 1/c

## Installation

### Requirements

- Python 3.10 or higher
- Required packages listed in `requirements.txt`

### Quick Start

```bash
# Clone the repository
git clone https://github.com/username/topological-liquidity-trap.git
cd topological-liquidity-trap

# Install dependencies
pip install -r requirements.txt

# Run the simulation
python simulation.py
```

## Usage

### Basic Simulation

```python
from simulation import run_monte_carlo_simulation

# Run with default parameters
results = run_monte_carlo_simulation(
    n_values=[500, 1000, 2000],
    n_trials=500,
    base_seed=42
)
```

### Custom Parameters

```python
import numpy as np

# Define custom activation probabilities
p_values = np.arange(0.10, 0.40, 0.02)

# Run simulation with custom parameters
results = run_monte_carlo_simulation(
    n_values=[1000],
    p_values=p_values,
    c=4.0,  # average degree (determines p_c = 0.25)
    n_trials=100,
    base_seed=42
)
```

### Computing Systemic Liquidity for a Single Graph

```python
from simulation import (
    generate_erdos_renyi_with_weights,
    simulate_site_percolation,
    compute_systemic_liquidity
)

# Generate a base graph
n = 1000
c = 4.0
G_base = generate_erdos_renyi_with_weights(n, c/n, mu=0.1, sigma=0.05, seed=42)

# Apply site percolation with 30% active agents
G_active = simulate_site_percolation(G_base, active_prob=0.30, seed=42)

# Compute systemic liquidity
threshold = 5 * np.log(n)  # l(n) = 5 log n
L_sys = compute_systemic_liquidity(G_active, n, threshold)
print(f"Systemic Liquidity: {L_sys:.4f}")
```

## Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `n_values` | List of network sizes to simulate | `[500, 1000, 2000]` |
| `p_values` | Array of activation probabilities | `np.arange(0.05, 0.525, 0.025)` |
| `c` | Base average degree | `4.0` |
| `n_trials` | Number of Monte Carlo trials per parameter | `500` |
| `base_seed` | Random seed for reproducibility | `42` |
| `mu` | Mean edge weight (positive for productive trades) | `0.1` |
| `sigma` | Standard deviation of edge weights | `0.05` |

## Output

The simulation returns a dictionary with results for each network size:

```python
results = {
    n: {
        'p_values': array of activation probabilities,
        'means': array of mean systemic liquidity values,
        'stds': array of standard deviations,
        'p_c': critical threshold (1/c)
    }
}
```

## Theoretical Background

### The Topological Liquidity Trap

When the fraction of active agents falls below the percolation threshold `p_c = 1/c`:

1. The Giant Strongly Connected Component (GSCC) disappears
2. All Strongly Connected Components have size `O(log n)`
3. The systemic liquidity metric converges to zero with high probability

### Key Formula

The thresholded systemic liquidity metric:

```
L_sys^(ℓ)(A) = Σ_C (|C|/n) × max(ρ(A_C), 0) × 1{|C| ≥ ℓ(n)}
```

where:
- `C` ranges over strongly connected components
- `ρ(A_C)` is the Max-Plus spectral radius
- `ℓ(n) = 5 log n` is the systemic threshold

## Reproducibility

All simulations use deterministic random seeds to ensure exact reproducibility. The seed for each trial is computed as:

```python
seed = base_seed + hash((n, p, trial)) % (2**31)
```

## Execution Time

| Network Size | Trials | Approximate Time |
|--------------|--------|------------------|
| n = 500 | 500 | ~20 minutes |
| n = 1000 | 500 | ~40 minutes |
| n = 2000 | 500 | ~90 minutes |

*Times measured on a standard laptop (Intel i7, 16GB RAM)*

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Author2026,
  title={Topological Phase Transitions in Over-the-Counter Markets: 
         A Structural View of Liquidity Collapse},
  author={Eshaghi, M. and Sorouhesh, M.R.},
  journal={Journal Name},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact:
- Email: babaxor@iau.ac.ir
- GitHub Issues: [Open an issue](https://github.com/Babaxor/topological-liquidity-trap/issues)
