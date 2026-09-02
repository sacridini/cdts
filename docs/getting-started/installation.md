# Installation

CDTS is designed to be easy to install and run across different environments. 

## Standard Installation

The easiest way to install CDTS is via Python's package manager, `pip`. We provide pre-compiled binaries (wheels) for Windows, macOS, and Linux, supporting Python 3.9 and newer. Because we distribute pre-compiled binaries, **you do not need a C++ compiler** installed on your machine for the standard installation.

```bash
pip install cdts
```

This command automatically installs all required Python dependencies, including `xarray`, `dask`, `scikit-learn`, `rasterio`, `torch`, and `pystac-client`.

## Optional Dependencies

For development and running tests, you can install the optional development dependencies:

```bash
pip install cdts[dev]
```

## Installing from Source

If you need to modify the C++ backend, use the latest unreleased features, or build the package on an unsupported architecture, you will need to install CDTS from the source.

### Requirements

*   Python 3.9+
*   A C++ Compiler supporting C++14 (GCC, Clang, or MSVC)

### Instructions

1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/sacridini/cdts.git
   cd cdts
   ```

2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

This process will invoke `pybind11` and your system's C++ compiler to build the core statistical engines (`src/main.cpp`, `src/landtrendr.cpp`, `src/ccdc.cpp`) and link them with the Eigen3 library.
