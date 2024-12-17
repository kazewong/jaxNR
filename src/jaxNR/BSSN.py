import jax
import jax.numpy as jnp
from jaxNR.data_structure import Evolver, Grid, Cell

class BSSN(Evolver):

    def evaluate_rhs():
        pass

    def evolve(self, grid: Grid) -> Grid:
        return grid