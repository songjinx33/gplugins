from __future__ import annotations

from gdsfactory.path import hashlib
from gdsfactory.pdk import LayerStack, get_layer_stack
from gdsfactory.typings import ComponentFactory
from gdsfactory import Component
from typing import Callable

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

    # Material map that maps materials from the layer_stack to the simulators' materials
    material_map: dict[str, str]

    def __init__(
        self,
        cell: ComponentFactory | Component,
        material_map: dict[str, str] = None,
        dependencies: list[dr.DesignRecipe] | None = None,
        layer_stack: LayerStack = get_layer_stack(),
    ):
        dependencies = dependencies or []
        self.dependencies = dr.ConstituentRecipes(dependencies)
        self.cell = cell
        self.last_hash = -1
        self.material_map = material_map
        self.layer_stack = layer_stack

    def __hash__(self) -> int:
        """
        Returns a hash of all state this DesignRecipe contains.
        Subclasses should include functionality-specific state (e.g. fdtd settings) here.
        This is used to determine 'freshness' of a recipe (i.e. if it needs to be rerun)
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

        # TODO find some way to automatically hook into eval()'s subclasses
        # and update last_hash  at the end? can maybe use decorators?
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
        # Evaluate the design recipe
        func(*args, **kwargs)
        # Evaluate independent recipes
        self = args[0]
        success = self.eval_dependencies()
        # Update hash
        self.last_hash = hash(self)
        # Return successful execution
        return success

    return design_recipe_eval
