import os

import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["JAX_ENABLE_X64"] = "True"

from dataclasses import dataclass
from functools import partial

from jax import jit, lax
from jax import numpy as jnp
from jax import scipy as jsp

from scd import hamiltonian, system


@dataclass
class ciss_tb(hamiltonian.ham_base):
    omega: float = 0.05 / 27.211385
    c: float = (2 * 0.1 * omega**3) ** 0.5
    tc: float = 0.02 / 27.211385
    lambda_soc: float = -1.0 * tc

    @partial(jit, static_argnums=(0, 1))
    def ham_mat(self, sys: system.system, nuc_pos: jnp.ndarray) -> jnp.ndarray:
        h = jnp.zeros((sys.n_states, sys.n_states)) + 0.0j
        t_vec = jnp.zeros(sys.n_sites)
        t_vec = t_vec.at[1].set(-self.tc)
        hopping = jsp.linalg.toeplitz(t_vec)
        eph = self.c * nuc_pos * jnp.eye(sys.n_sites)
        h = h.at[: sys.n_sites, : sys.n_sites].set(hopping + eph)
        h = h.at[sys.n_sites :, sys.n_sites :].set(hopping + eph)
        kappa = jnp.cos(sys.theta)
        tau = jnp.sin(sys.theta)

        # carry = h
        # x: site index
        def scanned_fun(carry, x):
            x_u = x
            x_d = x + sys.n_sites
            # uu
            carry = carry.at[x_u + 1, x_u].add(-1.0j * self.lambda_soc * sys.p * kappa)
            carry = carry.at[x_u, x_u + 1].add(1.0j * self.lambda_soc * sys.p * kappa)
            # dd
            carry = carry.at[x_d + 1, x_d].add(1.0j * self.lambda_soc * sys.p * kappa)
            carry = carry.at[x_d, x_d + 1].add(-1.0j * self.lambda_soc * sys.p * kappa)
            # ud
            carry = carry.at[x_u + 1, x_d].add(
                -1.0j
                * self.lambda_soc
                * (
                    sys.p * tau * jnp.sin((x + 0.5) * sys.dphi)
                    - 1.0j * tau * jnp.cos((x + 0.5) * sys.dphi)
                )
            )
            carry = carry.at[x_d, x_u + 1].add(
                1.0j
                * self.lambda_soc
                * (
                    sys.p * tau * jnp.sin((x + 0.5) * sys.dphi)
                    + 1.0j * tau * jnp.cos((x + 0.5) * sys.dphi)
                )
            )
            # du
            carry = carry.at[x_d + 1, x_u].add(
                -1.0j
                * self.lambda_soc
                * (
                    sys.p * tau * jnp.sin((x + 0.5) * sys.dphi)
                    + 1.0j * tau * jnp.cos((x + 0.5) * sys.dphi)
                )
            )
            carry = carry.at[x_u, x_d + 1].add(
                1.0j
                * self.lambda_soc
                * (
                    sys.p * tau * jnp.sin((x + 0.5) * sys.dphi)
                    - 1.0j * tau * jnp.cos((x + 0.5) * sys.dphi)
                )
            )
            return carry, x

        h, _ = lax.scan(scanned_fun, h, jnp.arange(sys.n_sites - 1))
        return h

    @partial(jit, static_argnums=(0, 1))
    def ham_ci_product(
        self, sys: system.system, ci: jnp.ndarray, nuc_pos: jnp.ndarray
    ) -> jnp.ndarray:
        eph = jnp.concatenate((self.c * nuc_pos, self.c * nuc_pos))
        kappa = jnp.cos(sys.theta)
        tau = jnp.sin(sys.theta)

        # carry = h * ci
        # x: site index
        def scanned_fun(carry, x):
            x_u = x
            x_d = x + sys.n_sites
            # hopping
            carry = carry.at[x_u].add(-self.tc * ci[x_u + 1])
            carry = carry.at[x_d].add(-self.tc * ci[x_d + 1])
            carry = carry.at[x_u + 1].add(-self.tc * ci[x_u])
            carry = carry.at[x_d + 1].add(-self.tc * ci[x_d])
            # soc
            # uu
            carry = carry.at[x_u].add(
                1.0j * self.lambda_soc * sys.p * kappa * ci[x_u + 1]
            )
            carry = carry.at[x_u + 1].add(
                -1.0j * self.lambda_soc * sys.p * kappa * ci[x_u]
            )
            # dd
            carry = carry.at[x_d].add(
                -1.0j * self.lambda_soc * sys.p * kappa * ci[x_d + 1]
            )
            carry = carry.at[x_d + 1].add(
                1.0j * self.lambda_soc * sys.p * kappa * ci[x_d]
            )
            # ud
            carry = carry.at[x_d].add(
                1.0j
                * self.lambda_soc
                * (
                    sys.p * tau * jnp.sin((x + 0.5) * sys.dphi)
                    + 1.0j * tau * jnp.cos((x + 0.5) * sys.dphi)
                )
                * ci[x_u + 1]
            )
            carry = carry.at[x_u + 1].add(
                -1.0j
                * self.lambda_soc
                * (
                    sys.p * tau * jnp.sin((x + 0.5) * sys.dphi)
                    - 1.0j * tau * jnp.cos((x + 0.5) * sys.dphi)
                )
                * ci[x_d]
            )
            # du
            carry = carry.at[x_u].add(
                1.0j
                * self.lambda_soc
                * (
                    sys.p * tau * jnp.sin((x + 0.5) * sys.dphi)
                    - 1.0j * tau * jnp.cos((x + 0.5) * sys.dphi)
                )
                * ci[x_d + 1]
            )
            carry = carry.at[x_d + 1].add(
                -1.0j
                * self.lambda_soc
                * (
                    sys.p * tau * jnp.sin((x + 0.5) * sys.dphi)
                    + 1.0j * tau * jnp.cos((x + 0.5) * sys.dphi)
                )
                * ci[x_u]
            )
            return carry, x

        ci_0 = 0.0 * ci
        h_ci, _ = lax.scan(scanned_fun, ci_0, jnp.arange(sys.n_sites - 1))
        h_ci += eph * ci
        return h_ci

    @partial(jit, static_argnums=(0, 1))
    def force(
        self, sys: system.system, ci: jnp.ndarray, nuc_pos: jnp.ndarray
    ) -> jnp.ndarray:
        f_class = -self.omega**2 * nuc_pos
        prob = (ci * ci.conjugate()).real
        f_elec = -self.c * (prob[: sys.n_sites] + prob[sys.n_sites :]).real
        return f_elec + f_class

    def __hash__(self):
        return hash(
            (
                self.omega,
                self.c,
                self.tc,
                self.lambda_soc,
            )
        )


if __name__ == "__main__":
    from scd import system

    np.set_printoptions(precision=6, suppress=True)
    ham = ciss_tb()
    sys = system.system()
    np.random.seed(0)
    nuc_pos, _ = sys.init_nuc(ham.omega)
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
