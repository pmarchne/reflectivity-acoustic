# FWI Inversion
Layered 1D Full Waveform Inversion using the **adjoint-state method**.

### Logic
* **Physics:** forward modeling with the reflectivity method.
* **Objective:** L2 misfit minimization.
* **Optimizer:** Scipy `L-BFGS-B`.
* **Parameters:** Inverts for layer $V_p$ (Velocity).

### Usage
Run the script to start the optimization:
```bash
python3 FWI.py