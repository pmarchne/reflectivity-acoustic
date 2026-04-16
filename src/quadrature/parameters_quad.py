class ParametersQuad:

    def __init__(
        self, nq_prop=128, nq_evan=64, psi_max=4.0, method="filon", free_surface=1
    ):
        self.free_surface = free_surface
        self.nq_prop = nq_prop
        self.nq_evan = nq_evan
        self.psi_max = psi_max
        self.method = method

    def __repr__(self):
        return f"test quad"
