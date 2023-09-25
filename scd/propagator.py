import os

import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["JAX_ENABLE_X64"] = "True"
from dataclasses import dataclass
from functools import partial

from jax import jit, lax
from jax import numpy as jnp

from scd import hamiltonian, system


@dataclass
class propagator:
    prop_time: float = 1.0  # ps
    dt_nuc: float = 50.0
    dt_e_steps_half: int = 50
    dt_e: float = None  # dt_nuc / dt_e_steps_half
    n_block_steps: int = 10  # steps in a block of propagation before measurement
    n_blocks: int = None  # time / (dt_nuc * n_block_steps)

    def __post_init__(self):
        self.dt_e = self.dt_nuc / self.dt_e_steps_half
        self.n_blocks = int(
            self.prop_time * 41342 / (self.dt_nuc * self.n_block_steps)
        )  # ps to au

    @partial(jit, static_argnums=(0, 1, 2))
    def propagate_elec(
        self,
        sys: system.system,
        ham: hamiltonian.ham_base,
        ci: jnp.ndarray,
        nuc_pos: jnp.ndarray,
    ) -> jnp.ndarray:
        """Propagate the electronic wavefunction using Runge-Kutta 4th order

        Args:
            system: system object
            ham: hamiltonian object
            ci: electronic wavefunction
            nuc_pos: nuclear positions

        Returns:
            ci: propagated electronic wavefunction
        """
        k1 = -1.0j * ham.ham_ci_product(sys, ci, nuc_pos)
        k2 = ci + 0.5 * self.dt_e * k1
        k2 = -1.0j * ham.ham_ci_product(sys, k2, nuc_pos)
        k3 = ci + 0.5 * self.dt_e * k2
        k3 = -1.0j * ham.ham_ci_product(sys, k3, nuc_pos)
        k4 = ci + self.dt_e * k3
        k4 = -1.0j * ham.ham_ci_product(sys, k4, nuc_pos)
        ci = ci + (self.dt_e / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return ci

    @partial(jit, static_argnums=(0, 1, 2))
    def propagate_elec_exact(
        self,
        sys: system.system,
        ham: hamiltonian.ham_base,
        ci: jnp.ndarray,
        nuc_pos: jnp.ndarray,
    ) -> jnp.ndarray:
        """Propagate the electronic wavefunction using exact diagonalization

        Args:
            system: system object
            ham: hamiltonian object
            ci: electronic wavefunction
            nuc_pos: nuclear positions

        Returns:
            ci: propagated electronic wavefunction
        """
        h = ham.ham_mat(sys, nuc_pos)
        e, u = jnp.linalg.eigh(h)
        ci = u @ (jnp.exp(-1.0j * e * self.dt_e) * (u.T.conj() @ ci))
        return ci

    @partial(jit, static_argnums=(0, 1, 2))
    def velocity_verlet(
        self, sys: system.system, ham: hamiltonian.ham_base, dat: dict
    ) -> dict:
        """Propagate the system using velocity verlet algorithm

        Args:
            system: system object
            ham: hamiltonian object
            dat: dictionary containing the state of the system

        Returns:
            dat: dictionary containing the propagated state of the system
        """
        v = dat["p"]
        force = dat["force"]

        # half electronic evolution
        # carry: dat
        def half_elec_evolution(carry, x):
            carry["ci"] = self.propagate_elec(sys, ham, carry["ci"], carry["r"])
            return carry, x

        dat, _ = lax.scan(half_elec_evolution, dat, jnp.arange(self.dt_e_steps_half))
        dat["ci"] /= jnp.sum(dat["ci"].conjugate() * dat["ci"])

        # nuclear evolution
        dat["r"] += v * self.dt_nuc + 0.5 * force * self.dt_nuc**2
        force_new = ham.force(sys, dat["ci"], dat["r"])
        v += 0.5 * (force + force_new) * self.dt_nuc
        dat["force"] = force_new
        dat["p"] = v

        # half electronic evolution
        dat, _ = lax.scan(half_elec_evolution, dat, jnp.arange(self.dt_e_steps_half))
        dat["ci"] /= jnp.sum(dat["ci"].conjugate() * dat["ci"])
        return dat

    @partial(jit, static_argnums=(0, 1, 2))
    def run_trajectory(
        self, sys: system.system, ham: hamiltonian.ham_base, dat: dict
    ) -> (dict, jnp.ndarray):
        """Run a single trajectory

        Args:
            system: system object
            ham: hamiltonian object
            dat: dictionary containing the state of the system

        Returns:
            dat: dictionary containing the propagated state of the system
            pop: populations of the electronic states (including t=0)
        """
        init_pop = (dat["ci"] * dat["ci"].conjugate()).real

        # one step of propagation
        # carry: dat
        def step_scan(carry, x):
            carry = self.velocity_verlet(sys, ham, carry)
            return carry, x

        # scan over blocks
        # carry: dat
        def block_scan(carry, x):
            carry, _ = lax.scan(step_scan, carry, jnp.arange(self.n_block_steps))
            pop = (carry["ci"] * carry["ci"].conjugate()).real
            return carry, pop

        dat["force"] = ham.force(sys, dat["ci"], dat["r"])
        dat, pop = lax.scan(block_scan, dat, jnp.arange(self.n_blocks))
        pop = jnp.concatenate((init_pop[None, :], pop), axis=0)
        return dat, pop

    def __hash__(self) -> int:
        return hash(
            (
                self.prop_time,
                self.dt_nuc,
                self.dt_e_steps_half,
                self.dt_e,
                self.n_block_steps,
                self.n_blocks,
            )
        )


if __name__ == "__main__":
    from scd import ciss_tb, system

    sys = system.system()
    ham = ciss_tb.ciss_tb()
    prop = propagator()
    dat = {}
    dat["r"], dat["p"] = sys.init_nuc(ham.omega)
    dat["ci"] = jnp.zeros((sys.n_states)) + 0.0j
    dat["ci"] = dat["ci"].at[0].set(1.0)

    ci = prop.propagate_elec(sys, ham, dat["ci"], dat["r"])
    ci_1 = prop.propagate_elec_exact(sys, ham, dat["ci"], dat["r"])
    print(np.allclose(ci, ci_1, atol=1e-4))

    dat["force"] = ham.force(sys, dat["ci"], dat["r"])
    dat = prop.velocity_verlet(sys, ham, dat)
    dat, pop = prop.run_trajectory(sys, ham, dat)
    print(pop.shape)
