import jax
import jax.numpy as jnp
from jaxNR.data_structure import Evolver, Grid, Cell

class BSSN(Evolver):

    def evaluate_rhs():
        pass

    def get_derivatives(self, grid: Grid) -> Grid:
        raise NotImplementedError

    def evolve(self, grid: Grid) -> Grid:
        raise NotImplementedError