from dataclasses import dataclass
import numpy as np


@dataclass
class Layer:
    """
    Represents a geological layer with physical properties.
    Attributes:
        h: thickness [m]
        vp: P-wave velocity [m/s]
        rho: density [kg/m^3]
    """
    h: float
    vp: float
    rho: float

    def __getitem__(self, index):
        return (self.h, self.vp, self.rho)[index]
    
    def __iter__(self):
        """Allow unpacking: h, vp, rho = layer"""
        return iter((self.h, self.vp, self.rho))

    def __repr__(self):
        return f"Layer(h={self.h:.1f}, vp={self.vp:.1f}, rho={self.rho:.1f})"


def create_layers(hs, vps, rhos):
    """
    Create a list of Layer objects from arrays.

    Args:
        hs: array of thicknesses
        vps: array of P-wave velocities
        rhos: array of densities

    Returns:
        list of Layer objects
    """
    hs = np.asarray(hs)
    vps = np.asarray(vps)
    rhos = np.asarray(rhos)

    if not len(hs) == len(vps) == len(rhos):
        raise ValueError("All arrays must have the same length")

    return [Layer(h, vp, rho) for h, vp, rho in zip(hs, vps, rhos)]


def to_arrays(layers):
    """
    Convert a list of Layer objects to NumPy arrays.

    Args:
        layers: list of Layer objects or tuples

    Returns:
        tuple of (hs, vps, rhos) as NumPy arrays
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


def update_layer(layers, index, h=None, vp=None, rho=None):
    """
    Update specific parameters of a layer at given index.

    Args:
        layers: list of Layer objects
        index: index of layer to update
        h: new thickness (optional)
        vp: new P-wave velocity (optional)
        rho: new density (optional)

    Returns:
        updated list of layers
    """
    layers = list(layers)  # Create a copy
    layer = layers[index]

    new_h = h if h is not None else layer.h
    new_vp = vp if vp is not None else layer.vp
    new_rho = rho if rho is not None else layer.rho

    layers[index] = Layer(new_h, new_vp, new_rho)
    return layers


def update_from_arrays(layers, hs=None, vps=None, rhos=None):
    """
    Update layers from modified arrays.

    Args:
        layers: list of Layer objects
        hs: updated thickness array (optional)
        vps: updated velocity array (optional)
        rhos: updated density array (optional)

    Returns:
        updated list of layers
    """
    # Get current arrays
    curr_hs, curr_vps, curr_rhos = to_arrays(layers)

    # Use updated arrays or keep current
    new_hs = hs if hs is not None else curr_hs
    new_vps = vps if vps is not None else curr_vps
    new_rhos = rhos if rhos is not None else curr_rhos

    return create_layers(new_hs, new_vps, new_rhos)


if __name__ == "__main__":
    print("=== Create layers from arrays ===")
    vps = np.array([1500., 2800., 3800., 2300., 6500.])
    hs = np.array([300.0, 350.0, 420.0, 620.0, 650.0])
    rhos = np.array([2000.0, 2000.0, 2000.0, 2000.0, 2000.0])

    layers = create_layers(hs, vps, rhos)
    print("Created layers:")
    for i, layer in enumerate(layers):
        print(f"  Layer {i}: {layer}")

    print("\n=== Convert back to arrays ===")
    h_arr, vp_arr, rho_arr = to_arrays(layers)
    print(f"h:   {h_arr}")
    print(f"vp:  {vp_arr}")
    print(f"rho: {rho_arr}")

    print("\n=== Update single layer ===")
    layers = update_layer(layers, index=1, vp=3500.)
    print(f"Updated layer 1: {layers[1]}")

    print("\n=== Update from modified arrays ===")
    vps[1] = 3650.
    vps[2] = 4500.
    layers = update_from_arrays(layers, vps=vps)
    print("Updated layers:")
    for i, layer in enumerate(layers):
        print(f"  Layer {i}: {layer}")

    print("\n=== Mixed usage ===")
    mixed_layers = [
        Layer(100.0, 1505.0, 2000.0),
        (200.0, 1613.0, 2000.0),  # tuple also works
        Layer(250.0, 1749.0, 2000.0),
    ]
    h, vp, rho = to_arrays(mixed_layers)
    print(f"From mixed input - vp: {vp}")
