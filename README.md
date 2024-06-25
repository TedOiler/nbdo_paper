# system_optidex

## Overview

`system_optidex` is a Python package designed to facilitate the Design of Experiments (DoE), focusing on optimizing and analyzing various mathematical models and experimental setups. It includes a suite of optimization algorithms, utility functions, mathematical models, and a user-friendly web application interface for conducting and managing experiments.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
    - [Web Application](#web-application)
    - [Optimization Algorithms](#optimization-algorithms)
    - [Mathematical Models](#mathematical-models)
    - [Utility Functions](#utility-functions)
    - [Testing and Experimentation](#testing-and-experimentation)
3. [Directory Structure](#directory-structure)
4. [Contributing](#contributing)
5. [License](#license)

## Installation

To install the package, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/system_optidex.git
cd system_optidex
pip install -r requirements.txt
```

## Usage

### Web Application

The package includes a web application built with Streamlit to provide a user-friendly interface for conducting experiments and visualizing results.

To run the web application:

```bash
streamlit run local_app/streamlit.py
```

### Optimization Algorithms

The `optimizers` directory contains various optimization algorithms designed for both continuous and discrete variables.

#### Example Usage

```python
from system_optidex.optimizers.cordex_continuous import CordexContinuous
optimizer = CordexContinuous()
result = optimizer.optimize(your_model, your_parameters)
```

### Mathematical Models

The `mathematical_models` directory includes predefined mathematical models used in the optimization process.

#### Example Usage

from system_optidex.mathematical_models.f_on_f import FOnFModel
model = FOnFModel()
response = model.evaluate(your_inputs)

### Utility Functions

The `utilities` directory contains utility functions for supporting calculations and analyses, such as basis functions, polynomial calculations, and statistical measures.

#### Example Usage

from system_optidex.utilities.optimalities import calculate_optimality
optimality = calculate_optimality(your_data)

### Testing and Experimentation

The `playground` directory includes Jupyter notebooks for experimenting with different aspects of the software. These notebooks provide hands-on examples and tests for various models and optimization techniques.

To run the notebooks:

```bash
cd playground/ae
jupyter notebook
```

## Directory Structure

The repository is organized as follows:

```bash
system_optidex/
│
├── README.md                  # Documentation
├── LICENSE                    # License information
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore file
│
├── local_app/                 # Streamlit web application
│   └── streamlit.py
│
├── playground/                # Jupyter notebooks for testing and experimentation
│   ├── ae/
│   ├── misc_test/
│   └── cordex/
│
├── optimizers/                # Optimization algorithms
│   ├── cordex_continuous.py
│   ├── cordex_discrete.py
│   ├── base_optimizer.py
│   ├── nbdo.py
│   └── __init__.py
│
├── utilities/                 # Utility functions
│   ├── criteria.py
│   ├── basis.py
│   ├── J_bsplines_poly.py
│   ├── calc_I_theta.py
│   ├── calc_Sigma.py
│   ├── J_step_poly.py
│   └── __init__.py
│
├── mathematical_models/       # Mathematical models
│   ├── f_on_f.py
│   ├── s_on_f.py
│   ├── base_model.py
│   └── __init__.py
│
└── .git/                      # Git version control directory
```

## Contributing

We welcome contributions from the community. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new `Pull Request`.

Please ensure that your code adheres to the project's coding standards.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
