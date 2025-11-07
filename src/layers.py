from dataclasses import dataclass
import numpy as np

@dataclass
class Layer:
    h: float    # thickness [m]
    vp: float   # P-wave velocity [m/s]
    rho: float  # density [kg/m^3]

def layers_to_arrays(layers):
    """
    Convert a list of Layer objects or (h, vp, rho) tuples 
    into NumPy arrays (h, vp, rho).
    """
    hs, vps, rhos = [], [], []
    for ly in layers:
        if isinstance(ly, Layer):
            h, vp, rho = ly.h, ly.vp, ly.rho
        else:
            h, vp, rho = ly
        hs.append(h)
        vps.append(vp)
        rhos.append(rho)
    return np.array(hs), np.array(vps), np.array(rhos)

if __name__ == "__main__":
    # Example usage
    layers = [
        Layer(100.0, 1505.0, 2000.0),
        (200.0, 1613.0, 2000.0),   # tuple also works
        Layer(250.0, 1749.0, 2000.0),
    ]

    h, vp, rho = layers_to_arrays(layers)
    print("h:", h)
    print("vp:", vp)
    print("rho:", rho)
