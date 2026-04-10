from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Layer:
    h: float
    vp: float
    rho: float

    def __getitem__(self, index):
        return (self.h, self.vp, self.rho)[index]

    def __iter__(self):
        return iter((self.h, self.vp, self.rho))

    def __repr__(self):
        return f"Layer(h={self.h:.1f}, vp={self.vp:.1f}, rho={self.rho:.1f})"


def interfaces_to_widths(z_interfaces):
    """
    Convert interface depths to layer thicknesses.

    Example:
        z_interfaces = [0, 100, 200, 350]
        widths = [100, 100, 150]
    """
    z_interfaces = np.asarray(z_interfaces, dtype=float)

    if z_interfaces.ndim != 1:
        raise ValueError("z_interfaces must be a 1D array")
    if len(z_interfaces) < 2:
        raise ValueError("z_interfaces must contain at least two values")
    if np.any(np.diff(z_interfaces) <= 0):
        raise ValueError("z_interfaces must be strictly increasing")

    return np.diff(z_interfaces)


def create_layers(hs, vps, rhos):
    """
    Create list of Layer from arrays of thicknesses, velocities, densities.
    """
    hs = np.asarray(hs, dtype=float)
    vps = np.asarray(vps, dtype=float)
    rhos = np.asarray(rhos, dtype=float)

    if not (len(hs) == len(vps) == len(rhos)):
        raise ValueError("hs, vps, rhos must have the same length")

    return [Layer(h, vp, rho) for h, vp, rho in zip(hs, vps, rhos)]


def create_layers_from_interfaces(z_interfaces, vps, rhos):
    """
    Build layers from interface depths and property arrays.
    z_interfaces must have length len(vps) + 1.
    """
    widths = interfaces_to_widths(z_interfaces)
    vps = np.asarray(vps, dtype=float)
    rhos = np.asarray(rhos, dtype=float)

    if not (len(widths) == len(vps) == len(rhos)):
        raise ValueError("Need len(z_interfaces)=len(vps)+1 and matching rhos")

    return create_layers(widths, vps, rhos)


def to_arrays(layers):
    """
    Convert a list of Layer objects or tuples to arrays: (hs, vps, rhos).
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

    out = (
        np.array(hs, dtype=float),
        np.array(vps, dtype=float),
        np.array(rhos, dtype=float),
    )
    return out


def update_layer(layers, index, h=None, vp=None, rho=None):
    """
    Update one layer at a given index and return a new list.
    """
    layers = list(layers)
    layer = layers[index]

    new_h = layer.h if h is None else h
    new_vp = layer.vp if vp is None else vp
    new_rho = layer.rho if rho is None else rho

    layers[index] = Layer(new_h, new_vp, new_rho)
    return layers


def update_from_arrays(layers, hs=None, vps=None, rhos=None):
    """
    Update all layers from optional arrays.
    """
    curr_hs, curr_vps, curr_rhos = to_arrays(layers)

    new_hs = curr_hs if hs is None else hs
    new_vps = curr_vps if vps is None else vps
    new_rhos = curr_rhos if rhos is None else rhos

    return create_layers(new_hs, new_vps, new_rhos)
