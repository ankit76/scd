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
class ciss_tb_1(hamiltonian.ham_base):
    omega: float = 0.05 / 27.211385
    c: float = (2 * 0.1 * omega**3) ** 0.5
    tc: float = 0.02 / 27.211385
    tc_1: float = 0.0 / 27.211385
    lambda_soc: float = -1.0 * tc
    v: float = 1.0 * tc
    lead_sites: int = 201

    @partial(jit, static_argnums=(0, 1))
    def ham_mat(self, sys: system.system, nuc_pos: jnp.ndarray) -> jnp.ndarray:
        h = jnp.zeros((sys.n_states, sys.n_states)) + 0.0j
        t_vec = jnp.zeros(sys.n_sites)
        t_vec = t_vec.at[1].set(-self.tc)
        t_vec = t_vec.at[sys.n_unit_sites].set(-self.tc_1)
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

        h_sys, _ = lax.scan(scanned_fun, h, jnp.arange(sys.n_sites - 1))

        lead_sites = self.lead_sites
        h_lead = jnp.zeros((2 * lead_sites, 2 * lead_sites)) + 0.0j
        t_vec = jnp.zeros(lead_sites)
        t_vec = t_vec.at[1].set(-self.tc)
        hopping = jsp.linalg.toeplitz(t_vec)
        h_lead = h_lead.at[:lead_sites, :lead_sites].set(hopping)
        h_lead = h_lead.at[
            lead_sites : 2 * lead_sites, lead_sites : 2 * lead_sites
        ].set(hopping)

        # putting together
        h = (
            jnp.zeros((sys.n_states + 4 * lead_sites, sys.n_states + 4 * lead_sites))
            + 0.0j
        )
        h = h.at[: 2 * lead_sites, : 2 * lead_sites].set(h_lead)
        h = h.at[
            2 * lead_sites : 2 * lead_sites + sys.n_states,
            2 * lead_sites : 2 * lead_sites + sys.n_states,
        ].set(h_sys)
        h = h.at[2 * lead_sites + sys.n_states :, 2 * lead_sites + sys.n_states :].set(
            h_lead
        )

        # lead system coupling at junctions
        # up
        h = h.at[lead_sites - 1, 2 * lead_sites].set(-self.v)
        h = h.at[2 * lead_sites, lead_sites - 1].set(-self.v)
        h = h.at[2 * lead_sites + sys.n_sites - 1, 2 * lead_sites + sys.n_states].set(
            -self.v
        )
        h = h.at[2 * lead_sites + sys.n_states, 2 * lead_sites + sys.n_sites - 1].set(
            -self.v
        )
        # dn
        h = h.at[2 * lead_sites - 1, 2 * lead_sites + sys.n_sites].set(-self.v)
        h = h.at[2 * lead_sites + sys.n_sites, 2 * lead_sites - 1].set(-self.v)
        h = h.at[
            2 * lead_sites + 2 * sys.n_sites - 1,
            2 * lead_sites + sys.n_states + lead_sites,
        ].set(-self.v)
        h = h.at[
            2 * lead_sites + sys.n_states + lead_sites,
            2 * lead_sites + 2 * sys.n_sites - 1,
        ].set(-self.v)
        return h

    @partial(jit, static_argnums=(0, 1))
    def ham_ci_product(
        self, sys: system.system, ci: jnp.ndarray, nuc_pos: jnp.ndarray
    ) -> jnp.ndarray:
        lead_sites = self.lead_sites
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
            # long range hopping
            # using jax behavior of not raising out of bound errors
            carry = carry.at[x_u].add(
                -self.tc_1
                * ci[x_u + sys.n_unit_sites]
                * (x - 2 * lead_sites < sys.n_sites - sys.n_unit_sites)
            )
            carry = carry.at[x_d].add(
                -self.tc_1
                * ci[x_d + sys.n_unit_sites]
                * (x - 2 * lead_sites < sys.n_sites - sys.n_unit_sites)
            )
            carry = carry.at[x_u + sys.n_unit_sites].add(
                -self.tc_1
                * ci[x_u]
                * (x - 2 * lead_sites < sys.n_sites - sys.n_unit_sites)
            )
            carry = carry.at[x_d + sys.n_unit_sites].add(
                -self.tc_1
                * ci[x_d]
                * (x - 2 * lead_sites < sys.n_sites - sys.n_unit_sites)
            )
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
        h_ci_sys, _ = lax.scan(
            scanned_fun, ci_0, jnp.arange(sys.n_sites - 1) + 2 * lead_sites
        )
        h_ci_sys = h_ci_sys.at[2 * lead_sites : 2 * lead_sites + sys.n_states].add(
            eph * ci[2 * lead_sites : 2 * lead_sites + sys.n_states]
        )

        def scanned_fun_leads(carry, x):
            up = x
            dn = x + lead_sites
            carry = carry.at[up].add(-self.tc * ci[up + 1])
            carry = carry.at[dn].add(-self.tc * ci[dn + 1])
            carry = carry.at[up + 1].add(-self.tc * ci[up])
            carry = carry.at[dn + 1].add(-self.tc * ci[dn])
            return carry, x

        h_ci_lead_left, _ = lax.scan(
            scanned_fun_leads, ci_0, jnp.arange(lead_sites - 1)
        )
        h_ci_lead_right, _ = lax.scan(
            scanned_fun_leads,
            ci_0,
            jnp.arange(lead_sites - 1) + 2 * lead_sites + sys.n_states,
        )

        # junctions
        h_ci_j = 0.0 * ci
        # followiing the convention of the ham_mat function
        # up
        h_ci_j = h_ci_j.at[lead_sites - 1].add(-self.v * ci[2 * lead_sites])
        h_ci_j = h_ci_j.at[2 * lead_sites].add(-self.v * ci[lead_sites - 1])
        h_ci_j = h_ci_j.at[2 * lead_sites + sys.n_sites - 1].add(
            -self.v * ci[2 * lead_sites + sys.n_states]
        )
        h_ci_j = h_ci_j.at[2 * lead_sites + sys.n_states].add(
            -self.v * ci[2 * lead_sites + sys.n_sites - 1]
        )
        # dn
        h_ci_j = h_ci_j.at[2 * lead_sites - 1].add(
            -self.v * ci[2 * lead_sites + sys.n_sites]
        )
        h_ci_j = h_ci_j.at[2 * lead_sites + sys.n_sites].add(
            -self.v * ci[2 * lead_sites - 1]
        )
        h_ci_j = h_ci_j.at[2 * lead_sites + 2 * sys.n_sites - 1].add(
            -self.v * ci[2 * lead_sites + sys.n_states + lead_sites]
        )
        h_ci_j = h_ci_j.at[2 * lead_sites + sys.n_states + lead_sites].add(
            -self.v * ci[2 * lead_sites + 2 * sys.n_sites - 1]
        )

        h_ci = h_ci_lead_left + h_ci_sys + h_ci_lead_right + h_ci_j

        return h_ci

    @partial(jit, static_argnums=(0, 1))
    def force(
        self, sys: system.system, ci: jnp.ndarray, nuc_pos: jnp.ndarray
    ) -> jnp.ndarray:
        f_class = -self.omega**2 * nuc_pos
        prob = (ci * ci.conjugate()).real
        prob = prob[2 * self.lead_sites : 2 * self.lead_sites + sys.n_states]
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
    ham = ciss_tb_1(tc_1=0.001)
    sys = system.system()
    np.random.seed(0)
    nuc_pos, _ = sys.init_nuc(ham.omega)
    h = ham.ham_mat(sys, nuc_pos)
    print(h.shape)
    print(np.allclose(h, h.T.conj()))
    ci = jnp.array(
        np.random.rand(sys.n_states + ham.lead_sites * 4)
        + 1.0j * np.random.rand(sys.n_states + ham.lead_sites * 4)
    )
    ci /= jnp.sqrt(jnp.sum(ci.conjugate() * ci))
    h_ci = h @ ci
    h_ci_1 = ham.ham_ci_product(sys, ci, nuc_pos)
    print(np.allclose(h_ci, h_ci_1))
    f = ham.force(sys, ci, nuc_pos)
    print(f.shape)
