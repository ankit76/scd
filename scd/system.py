import os

import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["JAX_ENABLE_X64"] = "True"
from dataclasses import dataclass
from typing import Any, Sequence

from jax import numpy as jnp


@dataclass
class system:
    n_unit_sites: int = 31
    n_units: int = 5
    n_orb: int = 1
    n_sites: int = None  # n_unit_sites * n_units
    n_states: int = None  # 2 * n_sites
    theta: float = 0.0 * np.pi / 20.0
    p: float = 1.0
    dphi: float = None  # p * 2 * np.pi / n_unit_sites
    beta: float = 1 / 0.00095

    def __post_init__(self):
        if self.n_sites is None:
            self.n_sites = self.n_unit_sites * self.n_units
        self.n_states = 2 * self.n_sites * self.n_orb
        if self.dphi is None:
            self.dphi = self.p * 2 * np.pi / self.n_unit_sites

    def init_nuc(self, omega: float) -> Sequence[Any]:
        """Initialize the nuclear positions and momenta

        Args:
            omega: nuclear frequency

        Returns:
            R: nuclear positions
            P: nuclear momenta
        """
        sigP = np.sqrt(omega / (2 * np.tanh(0.5 * self.beta * omega)))
        sigR = sigP / omega
        ndof = self.n_sites

        r = np.zeros((ndof))
        p = np.zeros((ndof))
        for d in range(ndof):
            r[d] = np.random.normal() * sigR
            p[d] = np.random.normal() * sigP
        return jnp.array(r), jnp.array(p)

    def __hash__(self):
        return hash((self.n_unit_sites, self.n_units, self.theta, self.p, self.beta))


if __name__ == "__main__":
    sys = system()
    r, p = sys.init_nuc(0.05)
    print(r.shape, p.shape)
