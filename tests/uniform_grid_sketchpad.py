import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree, Int

class BBH_BSSN:

    """
    A class for the simulation of binary black hole spacetime using the BSSN formalism.

    The ordering of fields is as follows:
    0: $\alpha$
    1-3: $\beta^i$
    4-6: $B^i$
    7: $\chi$
    8-10: $\Gamma^i$
    11: K
    12-17: $\gamma_{ij}$
    18-23: $A_{ij}$
    """

    mass1: Float
    mass2: Float
    distance: Float

    N_grid: Int
    field: Float[Array, " 24 N_grid N_grid N_grid"]

    def __init__(self,
        mass1: Float,
        mass2: Float,
        distance: Float,
        N_grid: Int
    ):
        self.mass1 = mass1
        self.mass2 = mass2
        self.distance = distance
        self.N_grid = N_grid

        self.field = self.generate_initial_condition(mass1, mass2, distance, N_grid)


    @staticmethod
    def generate_initial_condition(
        mass1: Float,
        mass2: Float,
        distance: Float,
        N_grid: Int
    ) -> Float[Array, " 24 N_grid N_grid N_grid"]:
        grid_coord = jnp.array(jnp.meshgrid(
            jnp.linspace(-1, 1, N_grid),
            jnp.linspace(-1, 1, N_grid),
            jnp.linspace(-1, 1, N_grid)
        ))
        BH1_location = jnp.array([0, 0, -distance/2])
        BH2_location = jnp.array([0, 0, distance/2])
        r1 = jnp.linalg.norm(grid_coord - BH1_location[:, None, None, None], axis=0)
        r2 = jnp.linalg.norm(grid_coord - BH2_location[:, None, None, None], axis=0)
        chi = (1 + mass1/2/r1 + mass2/2/r2)**(-1./4)
        field = jnp.zeros((24, N_grid, N_grid, N_grid))
        field = field.at[0].set(chi)
        field = field.at[7].set(chi)
        field = field.at[12].set(jnp.ones((N_grid, N_grid, N_grid)))
        field = field.at[15].set(jnp.ones((N_grid, N_grid, N_grid)))
        field = field.at[17].set(jnp.ones((N_grid, N_grid, N_grid)))
        return field

    def compute_derivatives():
        pass

    def computer_rhs():
        pass

    def forward_step():
        pass

    def simulate(self, n_steps: int):
        pass