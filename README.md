# gplugins 0.0.2

[![docs](https://github.com/gdsfactory/gplugins/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gplugins/)
[![PyPI](https://img.shields.io/pypi/v/gplugins)](https://pypi.org/project/plugins/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/plugins.svg)](https://pypi.python.org/pypi/plugins)
[![MIT](https://img.shields.io/github/license/gdsfactory/gplugins)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/main/gdsfactory)

gdsfactory plugins:

- `dagster` for data pipelines.
- `database` for simulation and measurement database.
- `devsim` TCAD device simulator.
- `meow` Eigen Mode Expansion (EME).
- `femwell` Finite Element Method Solver (heaters, modes, TCAD, RF waveguides).
- `gmsh` mesh structures.
- `tidy3d` Finite Difference Time Domain (FDTD) simulations on the cloud using GPU.
- `lumerical` For Ansys FDTD and Circuit interconnect.
- `kfactory` for fill, dataprep and testing.
- `ray` for distributed computing and optimization.
- `sax` S-parameter circuit solver.
- `schematic_editor`: for bokeh schematic editor.
- `meep` for FDTD.
- `mpb` for MPB mode solver.
- `web`: for gdsfactory webapp

## Installation

You can install all plugins with:

```
pip install "gplugins[database,devsim,femwell,gmsh,kfactory,meow,meshwell,ray,sax,tidy3d]" --upgrade
```

Or Install only the plugins you need `pip install gplugins[database,devsim]` from the available plugins:

Separate installation (not using pip):

- For Meep and MPB you need to use `conda` or `mamba` on MacOS, Linux or [Windows WSL (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/install) with `conda install pymeep=*=mpi_mpich_* -c conda-forge -y`

## Getting started

- [Read docs](https://gdsfactory.github.io/gplugins/)
- [Read gdsfactory docs](https://gdsfactory.github.io/gdsfactory/)
- [![Join the chat at https://gitter.im/gdsfactory-dev/community](https://badges.gitter.im/gdsfactory-dev/community.svg)](https://gitter.im/gdsfactory-dev/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
