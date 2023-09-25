import os
import time

import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ[
    "XLA_FLAGS"
] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false                            intra_op_parallelism_threads=1"
from functools import partial

# os.environ["JAX_ENABLE_X64"] = "True"
from typing import Any

from jax import numpy as jnp
from mpi4py import MPI

print = partial(print, flush=True)

from scd import hamiltonian, propagator, system

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def driver(
    sys: system.system,
    ham: hamiltonian.ham_base,
    prop: propagator.propagator,
    n_trajectories: int = 100,
    seed: int = 0,
    init_elec_state: Any = None,
    filename: str = "populations.dat",
) -> None:
    """Driver function for running multiple trajectories and averaging the results

    Args:
        sys: system object
        ham: hamiltonian object
        prop: propagator object
        n_trajectories: number of trajectories
        seed: random seed
        init_elec_state: initial electronic state
        filename: filename to save the populations
    """

    if rank == 0:
        print(f"# Number of cores: {size}\n")

    np.random.seed(seed + rank)
    dat = {}
    if init_elec_state is not None:
        dat["ci"] = jnp.array(init_elec_state)
    else:
        dat["ci"] = jnp.zeros((sys.n_states)) + 0.0j
        dat["ci"] = dat["ci"].at[0].set(1.0)
    populations = np.zeros((prop.n_blocks + 1, sys.n_states))
    wall_time = 0.0

    for i in range(n_trajectories):
        start_time = time.time()
        dat["r"], dat["p"] = sys.init_nuc(ham.omega)
        _, pop = prop.run_trajectory(sys, ham, dat)
        populations = populations + (np.array(pop) - populations) / (i + 1)
        wall_time += time.time() - start_time
        if rank == 0:
            if i % (max(n_trajectories // 10, 1)) == 0:
                print(f"Trajectory {i} done in {wall_time:.2f} seconds")

    global_populations = np.zeros((prop.n_blocks + 1, sys.n_states))
    comm.Reduce(
        [populations, MPI.DOUBLE], [global_populations, MPI.DOUBLE], op=MPI.SUM, root=0
    )
    if rank == 0:
        global_populations /= size
        print(f"Wall time: {wall_time:.2f} seconds")
        prop_times = np.arange(prop.n_blocks + 1) * prop.dt_nuc * prop.n_block_steps
        np.savetxt(filename, np.column_stack((prop_times, global_populations)))

    # broadcast populations to other ranks
    comm.barrier()
    comm.Bcast(global_populations, root=0)
    comm.barrier()
    return global_populations


if __name__ == "__main__":
    from scd import ciss_tb, system

    sys = system.system()
    ham = ciss_tb.ciss_tb()
    prop = propagator.propagator()
    driver(sys, ham, prop, n_trajectories=1, seed=0)
