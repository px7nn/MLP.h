# MLP.h

A tiny, single-header, dependency-free Multi-Layer Perceptron library for
C. Drop `MLP.h` into your project — no build system, no linking, no
external dependencies beyond the standard library.

**Version:** 0.7.0 · **License:** [MIT](LICENSE)


## Features

- Single header, C99/C11, no dependencies beyond `<stddef.h>`, `<stdlib.h>`, `<stdio.h>`,
  `<string.h>`, `<stdbool.h>`, and `<stdint.h>`.
- Arbitrary topologies via a plain `size_t[]` array.
- Configurable weight initialization (Xavier, He) per layer, automatic
  initializer selection based on activation, per-layer activation (linear,
  ReLU, leaky ReLU, sigmoid, tanh), and selectable loss (MSE, binary cross-entropy)
  via `NetworkConfig`.
- Full backpropagation + per-sample SGD training.
- Single-sample prediction via `MLP_Predict()` and batch dataset prediction
  via `MLP_Predict_Dataset()`.
- Model serialization — `MLP_Save_Network()` / `MLP_Load_Network()`
  round-trip a trained network to/from a file.
- CSV dataset loading — `MLP_LoadCSV()` reads a delimited file straight
  into a `Dataset`, with an optional header row and structured parse
  errors.
- Structured error handling via `MLP_GetLastError()` / `MLP_ErrorString()`,
  with an opt-in `MLP_EXIT_ON_ERROR` fail-fast mode for small programs.
- No hidden allocations you don't control by default — datasets built
  with `MLP_Create_Dataset` are just pointers into your own arrays.
  (`MLP_LoadCSV`-allocated datasets are the one exception — free those
  with `MLP_Destroy_Dataset`.)

## Quick start

```c
#define MLP_IMPLEMENTATION   // in exactly one .c file
#include "MLP.h"
```

See [`docs/getting_started.md`](docs/getting_started.md) for a full
walkthrough.


## Documentation

- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api.md)
- [Theory: how the forward pass, backprop, and training loop work](docs/theory.md)

See [`examples/`](examples/) for a full training example
(`xor_gate.c`), a companion example that loads the saved model back
in for inference (`load_model.c`), and an example that trains directly
from a CSV file via `MLP_LoadCSV` (`load_csv.c`).


## Versioning

`MLP_VERSION_STRING` (and the matching `_MAJOR`/`_MINOR`/`_PATCH` macros)
are defined at the top of `MLP.h`. This project is pre-1.0, so the public
API may still change between minor versions.

## License

MIT — see [LICENSE](LICENSE).