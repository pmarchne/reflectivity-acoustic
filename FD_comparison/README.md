# Seismic Finite Difference Datasets

Synthetic seismic data generated for benchmarking analytical reflectivity solutions against numerical finite difference (FD) methods.

## Simulation Parameters
| Parameter | Value |
| :--- | :--- |
| **Domain** | 700 x 700 m |
| **Source** | 10Hz Ricker (1st derivative) at (100m, 50m) |
| **Receivers** | 57 stations at z=75m (0 to 700m, every 12.5m) |
| **Time** | 1.024s duration |
| **Density** | Constant 2000 kg/m³ |

## Python Config
```python
config = Config(
    n_receivers=57, x_min=0., x_max=700.,
    z_rec=75., z_src=50., x_src=100., nq_prop=1024, f0=10.,
    total_time=1.024, delay=0.1, epsilon=1.0,
    source_deriv=True, free_surface=True,
)
```

## Dataset Scenarios

There are 4 variations (2 models × with/without free surface).

### Scenario 1
- **z_int**: `[0.0, 100.0, 200.0, 250.0, 350.0, 450.0, 550.0, 650.0, 700.0]`
- **Vp**: `[1505.0, 1603.0, 1749.0, 2019.0, 2179.0, 1900.0, 2265.0, 3281.0]`

### Scenario 2
- **z_int**: `[0.0, 100.0, 200.0, 275.0, 375.0, 400.0, 500.0, 550.0, 700.0]`
- **Vp**: `[1505.0, 2000.0, 2200.0, 4400.0, 1900.0, 3800.0, 2900.0, 3500.0]`

## Data Format
Raw `float32` binaries.
* **Velocity**: Reshape to `(141, 141)`
* **Seismograms**: Reshape to `(57, 2048)` &rarr; (Receivers, Time)