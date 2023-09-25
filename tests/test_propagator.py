import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ['JAX_ENABLE_X64'] = 'True'
import numpy as np
from jax import numpy as jnp

from scd import ciss_tb, propagator, system

seed = 92421
np.random.seed(seed)
sys = system.system(n_unit_sites=7, n_units=5)
ham = ciss_tb.ciss_tb()
prop = propagator.propagator()
dat = {}
dat["r"], dat["p"] = sys.init_nuc(ham.omega)
dat["ci"] = jnp.zeros((sys.n_states)) + 0.0j
dat["ci"] = dat["ci"].at[0].set(1.0)
dat["force"] = ham.force(sys, dat["ci"], dat["r"])


def test_propagate_elec():
    ci = prop.propagate_elec(sys, ham, dat["ci"], dat["r"])
    assert ci.shape == (sys.n_states,)
    ci_1 = prop.propagate_elec_exact(sys, ham, dat["ci"], dat["r"])
    assert ci_1.shape == (sys.n_states,)
    assert jnp.allclose(ci, ci_1, atol=1e-5)


def test_velocity_verlet():
    dat_1 = prop.velocity_verlet(sys, ham, dat)
    assert dat_1["r"].shape == (sys.n_sites,)
    assert dat_1["p"].shape == (sys.n_sites,)
    assert dat_1["force"].shape == (sys.n_sites,)
    assert dat_1["ci"].shape == (sys.n_states,)


def test_run_trajectory():
    dat_1, pop = prop.run_trajectory(sys, ham, dat)
    assert dat_1["r"].shape == (sys.n_sites,)
    assert dat_1["p"].shape == (sys.n_sites,)
    assert dat_1["force"].shape == (sys.n_sites,)
    assert dat_1["ci"].shape == (sys.n_states,)
    assert pop.shape == (prop.n_blocks + 1, sys.n_states)


if __name__ == "__main__":
    test_propagate_elec()
    test_velocity_verlet()
    test_run_trajectory()
