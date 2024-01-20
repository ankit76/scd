import os

import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["JAX_ENABLE_X64"] = "True"

from dataclasses import dataclass
from functools import partial
from typing import Any

from jax import jit
from jax import numpy as jnp

from scd import hamiltonian, system


@dataclass
class spin_boson(hamiltonian.ham_base):
    """
    Spin boson model

    parameters:
    omega: bath frequencies
    c: coupling strengths
    epsilon: bias
    delta: tunneling (diabatic coupling)
    """

    omega: tuple
    c: tuple
    epsilon: float = 0.0
    delta: float = 1.0

    @partial(jit, static_argnums=(0, 1))
    def ham_mat(self, sys: system.system, nuc_pos: jnp.ndarray) -> jnp.ndarray:
        # ignoring quadratic ho terms
        h = jnp.zeros((2, 2)) + 0.0j
        eph = jnp.sum(jnp.array(self.c) * nuc_pos)
        h = h.at[0, 0].set(self.epsilon + eph)
        h = h.at[1, 1].set(-self.epsilon - eph)
        h = h.at[0, 1].set(self.delta)
        h = h.at[1, 0].set(self.delta)
        return h

    @partial(jit, static_argnums=(0, 1))
    def ham_ci_product(
        self, sys: system.system, ci: jnp.ndarray, nuc_pos: jnp.ndarray
    ) -> jnp.ndarray:
        # corresposnding to the hamiltonian matrix in ham_mat
        h_ci = jnp.zeros_like(ci)
        eph = jnp.sum(jnp.array(self.c) * nuc_pos)
        h_ci = h_ci.at[0].set((self.epsilon + eph) * ci[0] + self.delta * ci[1])
        h_ci = h_ci.at[1].set((-self.epsilon - eph) * ci[1] + self.delta * ci[0])
        return h_ci

    @partial(jit, static_argnums=(0, 1))
    def force(
        self, sys: system.system, ci: jnp.ndarray, nuc_pos: jnp.ndarray
    ) -> jnp.ndarray:
        f_class = -jnp.array(self.omega) ** 2 * nuc_pos
        prob = (ci * ci.conjugate()).real
        f_elec = -jnp.array(self.c) * prob[0] + jnp.array(self.c) * prob[1]
        return f_class + f_elec

    def __hash__(self):
        return hash(
            (
                self.omega,
                self.c,
                self.epsilon,
                self.delta,
            )
        )


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    omega, c = system.debye_bath(100)
    ham = spin_boson(omega, c)
    sys = system.system(n_states=2)
    np.random.seed(0)
    nuc_pos, _ = sys.init_nuc(jnp.array(ham.omega))
    h = ham.ham_mat(sys, nuc_pos)
    print(h.shape)
    print(np.allclose(h, h.T.conj()))
    ci = jnp.array(np.random.rand(sys.n_states) + 1.0j * np.random.rand(sys.n_states))
    ci /= jnp.sqrt(jnp.sum(ci.conjugate() * ci))
    h_ci = h @ ci
    h_ci_1 = ham.ham_ci_product(sys, ci, nuc_pos)
    print(np.allclose(h_ci, h_ci_1))
    f = ham.force(sys, ci, nuc_pos)
    print(f.shape)
