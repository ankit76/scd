from abc import ABC, abstractmethod

from jax import numpy as jnp

from scd import system


class ham_base(ABC):
    @abstractmethod
    def ham_mat(self, sys: system.system, nuc_pos: jnp.ndarray) -> jnp.ndarray:
        """Construct the Hamiltonian matrix

        Args:
            sys: system object
            nuc_pos: nuclear positions

        Returns:
            h: Hamiltonian matrix
        """
        pass

    @abstractmethod
    def ham_ci_product(
        self, sys: system.system, ci: jnp.ndarray, nuc_pos: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the product of the Hamiltonian matrix with the electronic wavefunction

        Args:
            sys: system object
            ci: electronic wavefunction
            nuc_pos: nuclear positions

        Returns:
            h_ci: product of the Hamiltonian matrix with the electronic wavefunction
        """
        pass

    @abstractmethod
    def force(
        self, sys: system.system, ci: jnp.ndarray, nuc_pos: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the force on the nuclei

        Args:
            sys: system object
            ci: electronic wavefunction
            nuc_pos: nuclear positions

        Returns:
            force: force on the nuclei
        """
        pass
