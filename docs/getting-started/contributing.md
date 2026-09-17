# Contributing to CDTS

We welcome contributions from the community. Whether you are fixing bugs, improving documentation, or adding new features, your help is appreciated.

## Development Setup

To start contributing to the CDTS codebase, you will need to set up a local development environment. 

### Prerequisites

* Python 3.9+
* A C++ Compiler (GCC, Clang, or MSVC) for the core engine.

### Setup Instructions

1. **Fork and Clone:** Fork the repository on GitHub and clone your fork locally.
   ```bash
   git clone https://github.com/YOUR_USERNAME/cdts.git
   cd cdts
   ```

2. **Install in Editable Mode with Dev Dependencies:** Install the package so that changes to the Python code are immediately reflected without needing to reinstall. 
   ```bash
   pip install -e .[dev]
   ```
   *Note: Any changes made to the C++ source files (`src/*.cpp`) will require you to re-run the `pip install -e .` command to trigger a recompilation.*

## Testing

We use `pytest` for running our test suite. Ensure that all tests pass before submitting a pull request.

To run the tests with coverage reporting, execute:

```bash
pytest --cov=cdts tests/
```

## Building the Docs Locally

Documentation lives under `docs/` and is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). To preview your changes before opening a pull request:

```bash
pip install mkdocs-material
mkdocs serve
```

This starts a local server (by default at `http://127.0.0.1:8000`) that live-reloads as you edit any `.md` file or `mkdocs.yml`. Before opening a PR, also run `mkdocs build --strict` once — it fails on broken internal links and navigation errors that a plain `mkdocs build` (what CI runs) only logs as warnings.

## Pull Request Process

1. Create a new branch for your feature or bug fix.
2. Ensure your code follows professional Python standards.
3. Add tests for any new functionality in the `tests/` directory.
4. Update documentation if necessary.
5. Submit a pull request on GitHub, clearly describing the changes and referencing any related issues.
