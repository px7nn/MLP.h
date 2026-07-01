# MLP.h

A tiny, single-header, dependency-free Multi-Layer Perceptron library for
C. Drop `MLP.h` into your project — no build system, no linking, no
external dependencies beyond the standard library.

**Version:** 0.2.0 · **License:** [MIT](LICENSE)


## Features

- Single header, C99/C11, no dependencies beyond `<stdlib.h>`, `<stdio.h>`,
  and `<stdint.h>`.
- Arbitrary topologies via a plain `size_t[]` array.
- Leaky ReLU hidden activations, linear output (works for both
  regression and classification-by-thresholding).
- Full backpropagation + per-sample SGD training.
- Model serialization — `MLP_Save()` / `MLP_Load()` round-trip a trained
  network to/from a file.
- Structured error handling via `MLP_GetLastError()` / `MLP_ErrorString()`.
- No hidden allocations you don't control — datasets are just pointers
  into your own arrays.

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
(`xor_gate.c`) and a companion example that loads the saved model back
in for inference (`load_model.c`).


## Versioning

`MLP_VERSION_STRING` (and the matching `_MAJOR`/`_MINOR`/`_PATCH` macros)
are defined at the top of `MLP.h`. This project is pre-1.0, so the public
API may still change between minor versions.

## License

MIT — see [LICENSE](LICENSE).