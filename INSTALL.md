# Installation Guide

## Prerequisites

- Python ≥ 3.10
- `gfortran` with OpenMP support

```bash
# Debian/Ubuntu
sudo apt-get install -y gfortran
```

---

## Steps

### 1. Clone the repository

```bash
git clone https://github.com/pmarchne/reflectivity-acoustic.git
cd reflectivity-acoustic
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the package

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

This installs all Python dependencies and automatically compiles the Fortran extensions via `f2py`.

### 4. Install Jupyter and run the notebook

```bash
pip install jupyter
jupyter notebook notebooks/seismogram.ipynb
```

