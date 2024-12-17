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
    field: dict[str,Float[Array, " N_grid N_grid N_grid"]]

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
        self.keys = [
            "alpha",
            "beta1", "beta2", "beta3",
            "B1", "B2", "B3",
            "chi",
            "Gamma1", "Gamma2", "Gamma3",
            "K",
            "gamma11", "gamma12", "gamma13", "gamma22", "gamma23", "gamma33",
            "A11", "A12", "A13", "A22", "A23", "A33"
        ]


    def generate_initial_condition(
        self,
        mass1: Float,
        mass2: Float,
        distance: Float,
        N_grid: Int
    ) -> dict[str, Float[Array, " N_grid N_grid N_grid"]]:
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
        field = {}
        field["chi"] = chi
        field["alpha"] = chi
        field["gamma11"] = jnp.ones((N_grid, N_grid, N_grid))
        field["gamma22"] = jnp.ones((N_grid, N_grid, N_grid))
        field["gamma33"] = jnp.ones((N_grid, N_grid, N_grid))
        for key in self.keys:
            if key not in field:
                field[key] = jnp.zeros((N_grid, N_grid, N_grid))

        return field

    def compute_derivatives():
        pass

    @staticmethod
    def computer_rhs(field: Float[Array, " 24 N_grid N_grid N_grid"]) -> Float[Array, " 24 N_grid N_grid N_grid"]:
        result = jnp.zeros_like(field)
        result = result.at[0].set(-

    def forward_step():
        pass

    def simulate(self, n_steps: int):
        pass