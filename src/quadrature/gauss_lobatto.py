import numpy as np
from numpy.polynomial.legendre import Legendre

def gauss_lobatto_nodes(n):
    """
    Return Gauss-Lobatto-Legendre nodes on [-1,1].
    Includes endpoints -1 and 1.
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    # Legendre polynomial P_{n-1}
    P = Legendre.basis(n - 1)

    # derivative P'
    dP = P.deriv()

    # interior roots of derivative
    interior = dP.roots()

    # include endpoints
    nodes = np.concatenate(([-1.0], interior, [1.0]))

    # numerical safety: sort
    return np.sort(nodes)