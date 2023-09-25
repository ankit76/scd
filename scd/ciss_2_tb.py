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
class ciss_2_tb(hamiltonian.ham_base):
    """Two types of orbitals on each site indexed n and b, with sites ordered as n_u, b_u, n_d, b_d"""

    omega: float = 0.05 / 27.211385
    c: float = (2 * 0.1 * omega**3) ** 0.5
    tc: float = 0.02 / 27.211385
    lambda_soc: float = -1.0 * tc

    @partial(jit, static_argnums=(0, 1))
    def ham_mat(self, sys: system.system, nuc_pos: jnp.ndarray) -> jnp.ndarray:
        assert sys.n_orb == 2
        h = jnp.zeros((sys.n_states, sys.n_states)) + 0.0j
        t_vec = jnp.zeros(sys.n_sites)
        t_vec = t_vec.at[1].set(-self.tc)
        hopping = jsp.linalg.toeplitz(t_vec)
        eph = self.c * nuc_pos * jnp.eye(sys.n_sites)
        h = h.at[: sys.n_sites, : sys.n_sites].set(hopping + eph)
        h = h.at[sys.n_sites : 2 * sys.n_sites, sys.n_sites : 2 * sys.n_sites].set(
            hopping + eph
        )
        h = h.at[
            2 * sys.n_sites : 3 * sys.n_sites, 2 * sys.n_sites : 3 * sys.n_sites
        ].set(hopping + eph)
        h = h.at[3 * sys.n_sites :, 3 * sys.n_sites :].set(hopping + eph)
        kappa = jnp.cos(sys.theta)
        tau = jnp.sin(sys.theta)

        # carry = h
        # x: site index
        def scanned_fun(carry, x):
            n_u = x
            n_d = x + 2 * sys.n_sites
            b_u = x + sys.n_sites
            b_d = x + 3 * sys.n_sites
            # uu
            carry = carry.at[n_u, b_u].add(1.0j * self.lambda_soc * tau)
            carry = carry.at[b_u, n_u].add(-1.0j * self.lambda_soc * tau)
            # dd
            carry = carry.at[n_d, b_d].add(-1.0j * self.lambda_soc * tau)
            carry = carry.at[b_d, n_d].add(1.0j * self.lambda_soc * tau)
            # ud
            carry = carry.at[n_u, b_d].add(
                sys.p * self.lambda_soc * kappa * jnp.exp(-1.0j * sys.p * x * sys.dphi)
            )
            carry = carry.at[b_u, n_d].add(
                -sys.p * self.lambda_soc * kappa * jnp.exp(-1.0j * sys.p * x * sys.dphi)
            )
            # du
            carry = carry.at[b_d, n_u].add(
                sys.p * self.lambda_soc * kappa * jnp.exp(1.0j * sys.p * x * sys.dphi)
            )
            carry = carry.at[n_d, b_u].add(
                -sys.p * self.lambda_soc * kappa * jnp.exp(1.0j * sys.p * x * sys.dphi)
            )
            return carry, x

        h, _ = lax.scan(scanned_fun, h, jnp.arange(sys.n_sites))
        return h

    @partial(jit, static_argnums=(0, 1))
    def ham_ci_product(
        self, sys: system.system, ci: jnp.ndarray, nuc_pos: jnp.ndarray
    ) -> jnp.ndarray:
        assert sys.n_orb == 2
        eph = jnp.concatenate(
            (self.c * nuc_pos, self.c * nuc_pos, self.c * nuc_pos, self.c * nuc_pos)
        )
        kappa = jnp.cos(sys.theta)
        tau = jnp.sin(sys.theta)

        # carry = h * ci
        # x: site index
        def scanned_fun(carry, x):
            n_u = x
            b_u = x + sys.n_sites
            n_d = x + 2 * sys.n_sites
            b_d = x + 3 * sys.n_sites
            # hopping
            not_end_q = x < sys.n_sites - 1  # open boundary
            # NB: out of bounds index updates do not raise errors in jax
            # they do nothing, probably shouldn't rely on this
            carry = carry.at[n_u].add(-self.tc * ci[n_u + 1] * not_end_q)
            carry = carry.at[n_d].add(-self.tc * ci[n_d + 1] * not_end_q)
            carry = carry.at[b_u].add(-self.tc * ci[b_u + 1] * not_end_q)
            carry = carry.at[b_d].add(-self.tc * ci[b_d + 1] * not_end_q)
            carry = carry.at[n_u + 1].add(-self.tc * ci[n_u] * not_end_q)
            carry = carry.at[n_d + 1].add(-self.tc * ci[n_d] * not_end_q)
            carry = carry.at[b_u + 1].add(-self.tc * ci[b_u] * not_end_q)
            carry = carry.at[b_d + 1].add(-self.tc * ci[b_d] * not_end_q)
            # soc
            # uu
            carry = carry.at[n_u].add(1.0j * self.lambda_soc * tau * ci[b_u])
            carry = carry.at[b_u].add(-1.0j * self.lambda_soc * tau * ci[n_u])
            # dd
            carry = carry.at[n_d].add(-1.0j * self.lambda_soc * tau * ci[b_d])
            carry = carry.at[b_d].add(1.0j * self.lambda_soc * tau * ci[n_d])
            # ud
            carry = carry.at[n_u].add(
                sys.p
                * self.lambda_soc
                * kappa
                * jnp.exp(-1.0j * sys.p * x * sys.dphi)
                * ci[b_d]
            )
            carry = carry.at[b_u].add(
                -sys.p
                * self.lambda_soc
                * kappa
                * jnp.exp(-1.0j * sys.p * x * sys.dphi)
                * ci[n_d]
            )
            # du
            carry = carry.at[b_d].add(
                sys.p
                * self.lambda_soc
                * kappa
                * jnp.exp(1.0j * sys.p * x * sys.dphi)
                * ci[n_u]
            )
            carry = carry.at[n_d].add(
                -sys.p
                * self.lambda_soc
                * kappa
                * jnp.exp(1.0j * sys.p * x * sys.dphi)
                * ci[b_u]
            )
            return carry, x

        ci_0 = 0.0 * ci
        h_ci, _ = lax.scan(scanned_fun, ci_0, jnp.arange(sys.n_sites))
        h_ci += eph * ci
        return h_ci

    @partial(jit, static_argnums=(0, 1))
    def force(
        self, sys: system.system, ci: jnp.ndarray, nuc_pos: jnp.ndarray
    ) -> jnp.ndarray:
        assert sys.n_orb == 2
        f_class = -self.omega**2 * nuc_pos
        prob = (ci * ci.conjugate()).real
        f_elec = (
            -self.c
            * (
                prob[: sys.n_sites]
                + prob[sys.n_sites : 2 * sys.n_sites]
                + prob[2 * sys.n_sites : 3 * sys.n_sites]
                + prob[3 * sys.n_sites :]
            ).real
        )
        return f_class + f_elec

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
    ham = ciss_2_tb()
    sys = system.system(n_orb=2)
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
