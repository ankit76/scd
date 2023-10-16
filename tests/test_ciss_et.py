import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ['JAX_ENABLE_X64'] = 'True'
import numpy as np
from jax import numpy as jnp

from scd import ciss_et, system

seed = 92421
np.random.seed(seed)
omega, c = ciss_et.bath(100)
ham = ciss_et.ciss_et(omega, c)
sys = system.system(n_states=5)
nuc_pos = jnp.array(np.random.randn(len(omega)))
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
    assert f.shape == (nuc_pos.size,)


if __name__ == "__main__":
    test_ham_mat()
    test_force()
