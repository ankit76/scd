import os

import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["JAX_ENABLE_X64"] = "True"

from dataclasses import dataclass
from functools import partial
from typing import Any

from flax import struct
from jax import jit, lax
from jax import numpy as jnp
from jax import scipy as jsp

from scd import hamiltonian, system


def bath(n_points, omega_max=0.1 / 27.2114):
    omega = np.linspace(0.00000001, omega_max, n_points * 1000)
    gamma = 4.75e-05
    lam = 0.003675
    F = lambda x: (2 * lam / np.pi) * np.arctan(x / gamma)
    F_omega = F(omega)
    lam_s = F_omega[-1]

    omega_j = np.zeros((n_points))
    c_j = np.zeros((n_points))
    for i in range(n_points):
        j = i + 1
        omega_j[i] = omega[np.argmin(np.abs(F_omega - ((j - 0.5) / n_points) * lam_s))]
        c_j[i] = omega_j[i] * (2 * lam_s / n_points) ** 0.5
    return tuple(omega_j), tuple(c_j)


@dataclass
class ciss_et(hamiltonian.ham_base):
    """
    Hamiltonian for electron transfer model with CISS
    """

    # omega: jnp.ndarray = struct.field(pytree_node=False)
    # c: jnp.ndarray = struct.field(pytree_node=False)
    omega: tuple
    c: tuple
    ev: float = 0.037
    lam_1: float = 0.1 * ev
    lam_2: float = 0.2 * ev
    e_ct_1: float = -0.1 * ev + lam_1
    e_ct_2: float = -0.35 * ev + (lam_2**0.5 + lam_1**0.5) ** 2
    gamma_1: float = 0.5 * ev / 1000
    gamma_2: float = 0.25 * ev / 1000
    c2_c1: float = (lam_1**0.5 + lam_2**0.5) / lam_1**0.5
    j_exc: float = 0.116 * ev / 1000
    theta: float = np.pi / 16

    @partial(jit, static_argnums=(0, 1))
    def ham_mat(self, sys: system.system, nuc_pos: jnp.ndarray) -> jnp.ndarray:
        h = jnp.zeros((5, 5)) + 0.0j
        eph = jnp.sum(jnp.array(self.c) * nuc_pos)
        h = h.at[1, 1].set(self.e_ct_1 + self.j_exc + eph)
        h = h.at[2, 2].set(self.e_ct_1 - self.j_exc + eph)
        h = h.at[3, 3].set(self.e_ct_2 + self.j_exc + self.c2_c1 * eph)
        h = h.at[4, 4].set(self.e_ct_2 + self.j_exc + self.c2_c1 * eph)

        h = h.at[0, 1].set(self.gamma_1 * jnp.cos(self.theta))
        h = h.at[1, 0].set(self.gamma_1 * jnp.cos(self.theta))
        h = h.at[0, 2].set(-1.0j * self.gamma_1 * jnp.sin(self.theta))
        h = h.at[2, 0].set(1.0j * self.gamma_1 * jnp.sin(self.theta))
        h = h.at[1, 3].set(self.gamma_2)
        h = h.at[3, 1].set(self.gamma_2)
        h = h.at[2, 4].set(self.gamma_2)
        h = h.at[4, 2].set(self.gamma_2)
        return h

    @partial(jit, static_argnums=(0, 1))
    def ham_ci_product(
        self, sys: system.system, ci: jnp.ndarray, nuc_pos: jnp.ndarray
    ) -> jnp.ndarray:
        # corresposnding to the hamiltonian matrix in ham_mat
        h_ci = jnp.zeros_like(ci)
        eph_1 = jnp.sum(jnp.array(self.c) * nuc_pos)
        eph_2 = self.c2_c1 * eph_1
        h_ci = h_ci.at[0].set(
            self.gamma_1 * jnp.cos(self.theta) * ci[1]
            - 1.0j * self.gamma_1 * jnp.sin(self.theta) * ci[2]
        )
        h_ci = h_ci.at[1].set(
            (self.e_ct_1 + self.j_exc + eph_1) * ci[1]
            + self.gamma_1 * jnp.cos(self.theta) * ci[0]
            + self.gamma_2 * ci[3]
        )
        h_ci = h_ci.at[2].set(
            (self.e_ct_1 - self.j_exc + eph_1) * ci[2]
            + 1.0j * self.gamma_1 * jnp.sin(self.theta) * ci[0]
            + self.gamma_2 * ci[4]
        )
        h_ci = h_ci.at[3].set(
            (self.e_ct_2 + self.j_exc + eph_2) * ci[3] + self.gamma_2 * ci[1]
        )
        h_ci = h_ci.at[4].set(
            (self.e_ct_2 + self.j_exc + eph_2) * ci[4] + self.gamma_2 * ci[2]
        )
        return h_ci

    @partial(jit, static_argnums=(0, 1))
    def force(
        self, sys: system.system, ci: jnp.ndarray, nuc_pos: jnp.ndarray
    ) -> jnp.ndarray:
        f_class = -jnp.array(self.omega) ** 2 * nuc_pos
        prob = (ci * ci.conjugate()).real
        f_elec = -jnp.array(self.c) * (prob[1] + prob[2]) - self.c2_c1 * jnp.array(
            self.c
        ) * (prob[3] + prob[4])
        return f_class + f_elec

    def __hash__(self):
        return hash(
            (
                self.omega,
                self.c,
                self.ev,
                self.lam_1,
                self.lam_2,
                self.e_ct_1,
                self.e_ct_2,
                self.gamma_1,
                self.gamma_2,
                self.c2_c1,
                self.j_exc,
                self.theta,
            )
        )


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    omega, c = bath(100)
    ham = ciss_et(omega, c)
    sys = system.system(n_states=5)
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
