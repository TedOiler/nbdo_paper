# System Optidex

A comprehensive Python framework for optimal experimental design optimization, supporting various mathematical models and optimization algorithms.

## Overview

System Optidex is a research-oriented framework that implements advanced optimization algorithms for experimental design. It supports three main types of mathematical models:

- **Scalar-on-Scalar (S-on-S)**: Traditional regression models
- **Scalar-on-Function (S-on-F)**: Models where scalar responses depend on functional predictors
- **Function-on-Function (F-on-F)**: Models where both predictors and responses are functions

The framework includes multiple optimization algorithms:
- **Cordex Continuous**: Continuous optimization using coordinate descent
- **Cordex Discrete**: Discrete optimization for categorical factors
- **NBDO (Neural Bayesian Design Optimization)**: Deep learning-based optimization using autoencoders and Bayesian optimization

## Features

- **Multiple Mathematical Models**: Support for scalar and functional data models
- **Advanced Optimization Algorithms**: Both classical and machine learning-based approaches
- **Basis Function Support**: B-splines, polynomials, and Fourier bases
- **Flexible Design Criteria**: A-optimality, D-optimality, and custom objective functions
- **Visualization Tools**: Built-in plotting and design visualization
- **Extensible Architecture**: Easy to add new models and optimizers

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd system_optidex
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
system_optidex/
├── basis/                 # Basis function implementations
│   ├── bspline.py        # B-spline basis functions
│   ├── fourier.py        # Fourier basis functions
│   ├── polynomial.py     # Polynomial basis functions
│   └── basis.py          # Base class and utilities
├── mathematical_models/   # Mathematical model implementations
│   ├── base_model.py     # Abstract base class
│   ├── s_on_s.py         # Scalar-on-Scalar models
│   ├── s_on_f.py         # Scalar-on-Function models
│   └── f_on_f.py         # Function-on-Function models
├── optimizers/           # Optimization algorithms
│   ├── base_optimizer.py # Abstract base class
│   ├── cordex_continuous.py # Continuous coordinate descent
│   ├── cordex_discrete.py   # Discrete coordinate descent
│   └── nbdo.py           # Neural Bayesian Design Optimization
├── J/                    # J-matrix computations
│   └── jmatrix.py        # Integration matrix calculations
├── utilities/            # Helper functions and utilities
│   ├── plotting/         # Visualization tools
│   ├── criteria/         # Optimality criteria
│   └── help/            # Utility functions
├── playground/           # Example notebooks and scripts
│   ├── ae/              # Autoencoder examples
│   └── cordex/          # Cordex optimization examples
└── results/             # Output files and plots
```

## Quick Start

### Basic Usage Example

```python
import sys
import os
import numpy as np

# Add paths
sys.path.append(os.path.abspath("mathematical_models"))
sys.path.append(os.path.abspath("optimizers"))

from mathematical_models.s_on_s import ScalarOnScalarModel
from optimizers.cordex_continuous import CordexContinuous

# Create a scalar-on-scalar model
Kx = [5]  # Number of factors
s_on_s_model = ScalarOnScalarModel(Kx=Kx[0], order=2)

# Set up optimizer
N = 24  # Number of runs
optimizer = CordexContinuous(model=s_on_s_model, runs=N)

# Run optimization
best_design, best_objective_value = optimizer.optimize(
    epochs=10, 
    refinement_epochs=5
)

print(f"Best objective value: {best_objective_value}")
```

### Advanced Example with Functional Data

```python
from basis.bspline import BSplineBasis
from basis.polynomial import PolynomialBasis
from mathematical_models.s_on_f import ScalarOnFunctionModel
from optimizers.nbdo import NBDO

# Create basis functions
x_basis = BSplineBasis(degree=0, total_knots_num=4)
beta_basis = PolynomialBasis(degree=2)
bases_pairs = [(x_basis, beta_basis)]

# Create scalar-on-function model
s_on_f_model = ScalarOnFunctionModel(bases_pairs=bases_pairs)

# Set up NBDO optimizer
optimizer = NBDO(model=s_on_f_model, latent_dim=2)

# Generate training data
optimizer.compute_train_set(num_designs=500, runs=12)

# Train autoencoder
history = optimizer.fit(epochs=100, patience=10)

# Optimize design
best_criterion, best_design = optimizer.optimize(n_calls=50)
```

## Mathematical Models

### Scalar-on-Scalar (S-on-S)
Traditional regression models where both predictors and responses are scalar values. Supports polynomial terms up to specified order.

### Scalar-on-Function (S-on-F)
Models where scalar responses depend on functional predictors. Uses basis function expansions and integration matrices.

### Function-on-Function (F-on-F)
Models where both predictors and responses are functions. Supports complex functional relationships with regularization.

## Optimization Algorithms

### Cordex Continuous
Coordinate descent optimization for continuous design variables. Uses L-BFGS-B for local optimization within each coordinate.

### Cordex Discrete
Coordinate descent optimization for discrete/categorical design variables. Evaluates all possible levels for each coordinate.

### NBDO (Neural Bayesian Design Optimization)
Deep learning approach using:
- Autoencoder to learn design space structure
- Bayesian optimization in latent space
- Custom loss functions incorporating design criteria

## Basis Functions

The framework supports three types of basis functions:

- **B-splines**: Flexible piecewise polynomial functions
- **Polynomials**: Traditional polynomial basis
- **Fourier**: Trigonometric basis functions

## Examples and Tutorials

Check the `playground/` directory for comprehensive examples:

- `playground/ae/`: Autoencoder-based optimization examples
- `playground/cordex/`: Coordinate descent optimization examples
- `playground/J/`: J-matrix computation examples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{system_optidex,
  title={System Optidex: Optimal Experimental Design Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/system_optidex}
}
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Support

For questions and support, please open an issue on the GitHub repository. 