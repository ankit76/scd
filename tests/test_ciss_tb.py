import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ['JAX_ENABLE_X64'] = 'True'
import numpy as np
from jax import numpy as jnp

from scd import ciss_tb, system

seed = 92421
np.random.seed(seed)
sys = system.system(n_unit_sites=7, n_units=5)
ham = ciss_tb.ciss_tb(tc_1=0.001)
nuc_pos = jnp.array(np.random.randn(sys.n_sites))
ci = jnp.array(np.random.rand(sys.n_states) + 1.0j * np.random.rand(sys.n_states))
ci /= jnp.sqrt(jnp.sum(ci.conjugate() * ci))


def test_ham_mat():
    h = ham.ham_mat(sys, nuc_pos)
    assert h.shape == (sys.n_states, sys.n_states)
    assert jnp.allclose(h, h.T.conj())
    h_ci = h @ ci
    assert h_ci.shape == (sys.n_states,)
    h_ci_1 = ham.ham_ci_product(sys, ci, nuc_pos)
    assert h_ci_1.shape == (sys.n_states,)
    assert jnp.allclose(h_ci, h_ci_1)


def test_force():
    f = ham.force(sys, ci, nuc_pos)
    assert f.shape == (sys.n_sites,)


if __name__ == "__main__":
    test_ham_mat()
    test_force()
