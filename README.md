# Code for "Safe Bayesian Optimization for Uncertain Correlations Matrices in Linear Models of Co-Regionalization"

This repository contains the source code and supplementary material for the paper:

> Jannis O. Lübsen and Annika Eichler, "Safe Bayesian Optimization for Uncertain Correlations Matrices in Linear Models of Co-Regionalization"
> Submitted to the IFAC World Congress 2026.

The code allows for the reproduction of all tables and figures presented in the manuscript.

# Safe Bayesian Optimization for Uncertain Correlations Matrices

This repository contains the code required to reproduce all tables and figures presented in the manuscript: **"Safe Bayesian Optimization for Uncertain Correlations Matrices in Linear Models of Coregionalization."**

## Prerequisites

The codebase was developed and tested in the following environment:
* **OS:** Ubuntu 24.04.2 LTS
* **Language:** Python 3.12.8

## Installation

To set up the environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone 
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd ./2025-code-safe-bayesian-optimization-for-uncertain-correlations-matrices-in-linear-models-of-coregionalization
    ```

3.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Reproducing Results

### 1. Running the Optimization Algorithms

**Multitask Bayesian Optimization:**
To run the multitask algorithm, execute the `rkhs_opt.py` script. You can specify the model type (`ICM` or `LMC`) using the `model_type` parameter inside the `rkhs_opt.py` script (default is `LMC`).

```bash
# Run with default LMC model
python rkhs_opt.py
```

**Single Task Bayesian Optimization:**
For the single-task case, execute the `rkhs_opt.py` script.

```bash
python rkhs_opt_ST.py
```

### 2. Generating Plots ###

To generate the plots used in the manuscript:
1. Navigate to the `plot_scripts` directory
2. Open and run the Jupyter Notebook `generate_plots.ipynb`
