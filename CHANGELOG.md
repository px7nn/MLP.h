# Changelog

All notable changes to this project will be documented in this file.

## [0.6.0] - 2026-07-07

### Added
- Single-sample prediction function `MLP_Predict(const Network *net, double *input, double *output)`.
- Support for configuring weight initializers per layer via `NetworkConfig.initializers` (an array of initializers of size `topology_size - 1`).
- Automatic selection of the best initializer for each layer based on its activation using `MLP_AUTO_INITIALIZERS` (mapped to `NULL`).
- Internal helper `_get_best_initializer(Activation act)` mapping `ACT_RELU` and `ACT_LEAKY_RELU` to `INIT_HE`, and `ACT_SIGMOID` and `ACT_LINEAR` to `INIT_XAVIER`.
- New visualization example `examples/visual_sin.c` which fits a sine wave and demonstrates interpolation vs. extrapolation.

### Changed
- Removed `INIT_RANDOM` from the `Initializer` enum.
- Updated `NetworkConfig` struct to take `const Initializer *initializers` instead of a single `const Initializer initializer` field.

### Docs
- Updated `README.md`, `docs/getting_started.md`, `docs/api.md`, and `docs/theory.md` to reflect version `0.6.0` changes including per-layer initialization and the single-sample `MLP_Predict()` API.

---

## [0.5.0] - 2026-07-06

### Added
- Support for different weight initializers via `NetworkConfig.initializer`:
  - `INIT_RANDOM` (uniform random weights in `[-1, 1]`)
  - `INIT_XAVIER` (Xavier initialization scaled by fan-in)
  - `INIT_HE` (He initialization scaled by fan-in)
- New `Initializer` enum and `INIT_COUNT` sentinel.
- Internal math helper `_sqrt()` using Newton's method for calculating weight scales without including `math.h`.
- Validation checks for initializers in `MLP_Create_Network`.

### Changed
- Updated default example files `examples/load_csv.c` and `examples/xor_gate.c` to use `INIT_HE` weight initialization.

### Docs
- Updated `README.md`, `docs/getting_started.md`, `docs/api.md`, and `docs/theory.md` to reflect version `0.5.0` and the new weight initialization options.

---

## [0.4.0] - 2026-07-03

### Added
- Configurable per-layer activations via `NetworkConfig.activations`:
  `ACT_LINEAR`, `ACT_RELU`, `ACT_LEAKY_RELU`, `ACT_SIGMOID`.
- Selectable training loss via `NetworkConfig.loss`: `LOSS_MSE`,
  `LOSS_BINARY_CROSS_ENTROPY`. `MLP_Train`'s output-layer delta now
  branches on `net->loss`, using the simplified `pred - target`
  gradient for `LOSS_BINARY_CROSS_ENTROPY` (intended for a sigmoid
  output).

### Changed
- **Breaking:** `MLP_Create_Network` now takes a single
  `const NetworkConfig *cfg` (topology, activations, loss) instead of
  a raw `(const size_t *topology, size_t n_layers)` pair, replacing
  the previous hardcoded leaky-ReLU-hidden/linear-output/MSE network.
- Saved model files now also store each layer's `Activation`, so
  `MLP_VERSION` (the on-disk format version) was bumped from `1` to
  `2`. A `0.3.0` model file will fail to load with
  `MLP_ERR_FILE_FORMAT` — retrain and re-save it.

### Docs
- Updated `README.md`, `docs/api.md`, and `docs/getting_started.md`
  for the `0.4.0` `NetworkConfig`-based API.

---

## [0.3.0] - 2026-07-02

### Added
- CSV dataset loading with `MLP_LoadCSV()` — reads a delimited file
  straight into a heap-allocated `Dataset` (optional header row,
  configurable feature/output column counts, row-count cap).
- `MLP_Destroy_Dataset()` to free a `Dataset` allocated by `MLP_LoadCSV()`.
- Opt-in `MLP_EXIT_ON_ERROR` macro: define it before including `MLP.h`
  to have any failing public API call print the error and `exit(EXIT_FAILURE)`
  immediately, instead of just returning `false`/a zeroed struct.
- `MLP_CSV_LINE_BUFFER` macro (default `1024`) controlling the size of
  the internal line buffer used while parsing CSV files.
- New `MLP_Error` codes for CSV parsing failures: `MLP_ERR_CSV_EMPTY`,
  `MLP_ERR_CSV_INVALID_NUMBER`, `MLP_ERR_CSV_COLUMN_COUNT`,
  `MLP_ERR_CSV_LINE_TOO_LONG`, `MLP_ERR_CSV_HEADER`.
- `datasets/circle.csv` sample dataset and `examples/load_csv.c`,
  demonstrating `MLP_LoadCSV()`, training, prediction, and cleanup with
  `MLP_Destroy_Dataset()`.

### Changed
- **Breaking:** renamed `MLP_Save()`/`MLP_Load()` to `MLP_Save_Network()`/
  `MLP_Load_Network()` for symmetry with the new `MLP_Destroy_Dataset()`
  and to disambiguate from dataset I/O now that the library loads data
  too.
- **Breaking:** `Dataset.samples`/`Dataset.output` are no longer
  `const`-qualified, so datasets returned by `MLP_LoadCSV()` (and
  destroyed by `MLP_Destroy_Dataset()`) can be freed internally by the
  library; `MLP_Create_Dataset()` still accepts plain non-const pointers
  as before.

### Docs
- Updated `README.md`, `docs/api.md`, and `docs/getting_started.md` for
  the `0.3.0` API: version macros, `MLP_Save_Network`/`MLP_Load_Network`
  naming, `MLP_LoadCSV`/`MLP_Destroy_Dataset`, `MLP_EXIT_ON_ERROR`,
  `MLP_CSV_LINE_BUFFER`, and the new CSV error codes.

---

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