# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-07-02

### Added
- Model serialization with `MLP_Save()` and `MLP_Load()`.
- Error handling API with `MLP_GetLastError()` and `MLP_ErrorString()`.
- Internal `_mlp_set_error()` helper.
- Model file format identifiers:
    - `MLP_MAGIC`
    - `MLP_VERSION`

### Changed
- Added `<stdint.h>` dependency for fixed-width integer types used by model serialization.
- Improved error reporting across public API functions.
- Save/Load operations now report detailed errors for file I/O, allocation failures, and invalid model formats.

### Examples
- Added `load_model.c` demonstrating how to load a saved model and perform inference.
- Updated `xor_gate.c` to demonstrate model serialization and improved error handling.

---

## [0.1.0] - 2026-07-01

Initial public release of **MLP.h**, a tiny single-header Multi-Layer Perceptron library for C.

## Features
- Single-header, dependency-free implementation
- Configurable feedforward network topology
- ReLU hidden activation with linear output
- Gradient descent training using backpropagation
- Dataset abstraction for training and inference
- Batch prediction support
- Network and dataset inspection utilities
- MIT licensed

This is the first public release and the API may evolve in future versions as additional functionality and improvements are introduced.