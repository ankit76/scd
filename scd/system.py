import os

import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["JAX_ENABLE_X64"] = "True"
from dataclasses import dataclass
from typing import Any, Sequence, Union

from jax import numpy as jnp


@dataclass
class system:
    n_unit_sites: int = 31
    n_units: int = 5
    n_orb: int = 1
    n_sites: int = None  # n_unit_sites * n_units, these only refer to the system
    n_states: int = None  # 2 * n_sites, these only refer to the system
    theta: float = 0.0 * np.pi / 20.0
    p: float = 1.0
    dphi: float = None  # p * 2 * np.pi / n_unit_sites
    beta: float = 1 / 0.00095
    n_lead_sites: int = 0
    n_lead_phonon_sites: int = 0

    def __post_init__(self):
        if self.n_sites is None:
            self.n_sites = self.n_unit_sites * self.n_units
        if self.n_states is None:
            self.n_states = 2 * self.n_sites * self.n_orb
        if self.dphi is None:
            self.dphi = self.p * 2 * np.pi / self.n_unit_sites

    # TODO: define separate function for et model
    def init_nuc(self, omega: Union[float, jnp.ndarray]) -> Sequence[Any]:
        """Initialize the nuclear positions and momenta

        Args:
            omega: nuclear frequency

        Returns:
            R: nuclear positions
            P: nuclear momenta
        """
        if isinstance(omega, tuple):
            omega = jnp.array(omega)
        sigP = np.sqrt(omega / (2 * np.tanh(0.5 * self.beta * omega)))
        sigR = sigP / omega
        n_dof = self.n_sites + 2 * self.n_lead_phonon_sites
        # if omega is an array
        if isinstance(omega, jnp.ndarray):
            n_dof = omega.size

        r = np.zeros((n_dof))
        p = np.zeros((n_dof))
        for d in range(n_dof):
            if isinstance(omega, jnp.ndarray):
                r[d] = np.random.normal() * sigR[d]
                p[d] = np.random.normal() * sigP[d]
            else:
                r[d] = np.random.normal() * sigR
                p[d] = np.random.normal() * sigP
        return jnp.array(r), jnp.array(p)

    def __hash__(self):
        return hash(
            (
                self.n_unit_sites,
                self.n_units,
                self.theta,
                self.p,
                self.beta,
                self.n_lead_sites,
                self.n_lead_phonon_sites,
            )
        )


if __name__ == "__main__":
    sys = system()
    r, p = sys.init_nuc(0.05)
    print(r.shape, p.shape)
