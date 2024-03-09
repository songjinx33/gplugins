from __future__ import annotations

from collections.abc import Callable

from gdsfactory import Component
from gdsfactory.path import hashlib
from gdsfactory.pdk import LayerStack, get_layer_stack
from gdsfactory.typings import ComponentFactory
from pathlib import Path

import pickle
import gplugins.design_recipe as dr


class DesignRecipe:
    """
    A DesignRecipe represents a flow of operations on GDSFactory components,
    with zero or more dependencies. Note that dependencies are assumed to be independent,
    dependent dependencies should be nested. When `eval()`ed, A DesignRecipe `eval()`s
    its dependencies if they they've become stale,
    and optionally executes some tool-specific functionality.
    For example,an FdtdDesignRecipe might simulate its `component` in
    Lumerial FDTD to extract its s-parameters.
    """

    # This `DesignRecipe`s dependencies. These are assumed to be independent
    dependencies: dr.ConstituentRecipes

    # the hash of the system last time eval() was executed
    last_hash: int

    # The component factory this DesignRecipe operates on. This is not necessarily
    # the same `component` referred to in the `dependencies` recipes.
    cell: ComponentFactory | Component | None = None

    # LayerStack for the process that the component is generated for
    layer_stack: LayerStack

    # Run convergence if True. Accurate simulations come from simulations that have run convergence.
    run_convergence: bool = True

    def __init__(
        self,
        cell: ComponentFactory | Component,
        dependencies: list[dr.DesignRecipe] | None = None,
        layer_stack: LayerStack = get_layer_stack(),
    ):
        dependencies = dependencies or []
        self.dependencies = dr.ConstituentRecipes(dependencies)
        self.cell = cell
        self.last_hash = -1
        self.layer_stack = layer_stack

    def __hash__(self) -> int:
        """
        Returns a hash of all state this DesignRecipe contains.
        Subclasses should include functionality-specific state (e.g. fdtd settings) here.
        This is used to determine 'freshness' of a recipe (i.e. if it needs to be rerun)

        Hashed items:
        - component or cell
        - layer stack
        """
        h = hashlib.sha1()
        if self.cell is not None:
            if isinstance(self.cell, Callable):
                h.update(self.cell().hash_geometry(precision=1e-4).encode("utf-8"))
            elif type(self.cell) == Component:
                h.update(self.cell.hash_geometry(precision=1e-4).encode("utf-8"))
        h.update(self.layer_stack.model_dump_json().encode("utf-8"))
        return int.from_bytes(h.digest(), "big")

    def is_fresh(self) -> bool:
        """
        Returns if this DesignRecipe needs to be re-`eval()`ed.
        This could be either caused by this DesignRecipe's
        configuration being changed, or that of one of its dependencies.
        """
        return hash(self) == self.last_hash and all(
            recipe.is_fresh() for recipe in self.dependencies
        )

    def eval(self, force_rerun_all=False) -> bool:
        """
        Evaluate this DesignRecipe. This should be overridden by
        subclasses with their specific functionalities
        (e.g. run the fdtd engine).
        Here we only evaluate dependencies,
        since the generic DesignRecipe has no underlying task.
        """
        success = self.eval_dependencies(force_rerun_all)

        self.last_hash = hash(self)
        return success

    def eval_dependencies(self, force_rerun_all=False) -> bool:
        """
        Evaluate this `DesignRecipe`'s dependencies.
        Because `dependencies` are assumed to be independent,
        they can be evaluated in any order.
        """
        success = True
        for recipe in self.dependencies:
            if force_rerun_all or (not recipe.is_fresh):
                success = success and recipe.eval(force_rerun_all)
        return success

class RecipeResults:
    """
    Design recipe results are stored in this dynamic class.

    This class allows designers to arbitrarily add results. Results are pickled to be saved onto working system.
    Results can be retrieved via unpickling.
    """

    def __init__(self, dirpath: Path | None = None, **kwargs):
        if isinstance(dirpath, str):
            dirpath = Path(dirpath)
        self.dirpath = dirpath or Path(__file__).resolve().parent
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save_pickle(self, dirpath: Path | None = None):
        """
        Save results by pickling as 'recipe_results.pkl' file

        Parameters:
            dirpath: Directory to store pickle file
        """
        if dirpath == None:
            with open(str(self.dirpath / 'recipe_results.pkl'), 'wb') as f:
                pickle.dump(self, f)
        else:
            with open(str(dirpath / 'recipe_results.pkl'), 'wb') as f:
                pickle.dump(self, f)

    def get_pickle(self, dirpath: Path | None = None) -> object:
        """
        Get results from 'recipe_results.pkl' file

        Parameters:
            dirpath: Directory to get pickle file

        Returns:
            RecipeResults as an object with results
        """
        if isinstance(dirpath, str):
            dirpath = Path(dirpath)
        if dirpath == None:
            with open(str(self.dirpath / 'recipe_results.pkl'), 'rb') as f:
                results = pickle.load(f)
        else:
            with open(str(dirpath / 'recipe_results.pkl'), 'rb') as f:
                results = pickle.load(f)

        return results

    def available(self, dirpath: Path | None = None) -> bool:
        """
        Check if 'recipe_results.pkl' file exists and results can be loaded

        Parameters:
            dirpath: Directory with pickle file

        Returns:
            True if results exist, False otherwise.
        """
        if isinstance(dirpath, str):
            dirpath = Path(dirpath)
        if dirpath == None:
            results_file = self.dirpath / 'recipe_results.pkl'
        else:
            results_file = dirpath / 'recipe_results.pkl'
        return results_file.is_file()

def eval_decorator(func):
    """
    Design recipe eval decorator

    Parameters:
        func: Design recipe eval method

    Returns:
        Design recipe eval method decorated with dependency execution and hashing
    """

    def design_recipe_eval(*args, **kwargs):
        """
        Evaluates design recipe and its dependencies then hashes the design recipe and returns successful execution
        """
        self = args[0]
        if "run_convergence" in kwargs:
            self.run_convergence = kwargs["run_convergence"]
        self.last_hash = hash(self)
        # Check if results already available. Results must be stored in directory with the same hash.

        # Evaluate the design recipe
        func(*args, **kwargs)
        # Evaluate independent recipes
        success = self.eval_dependencies()
        # Update hash
        self.last_hash = hash(self)
        # Return successful execution
        return success

    return design_recipe_eval
