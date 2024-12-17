import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Cell(ABC):
    pass

@dataclass
class Grid(ABC):
    n_dim: int
    data: PyTree[Cell]

@dataclass
class StaticGrid(Grid):
    n_res: list[int]

class Evolver(ABC):
    """
    An abstract class for the evolution of a grid.
    Thinking forward about the possibility of having ML evolver,
    this interface should be relatively general, such that it would take in a grid of cells,
    run some computation, then return a new field.
    """

    @abstractmethod
    def evolve(self, grid: Grid) -> Grid:
        pass

class Simulation:

    config: dict[str, Float]
    grid: Grid
    evolver: list[Evolver]