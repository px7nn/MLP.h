# API Reference

All public symbols are declared in `MLP.h`. Implementations are only
compiled in the translation unit that defines `MLP_IMPLEMENTATION` before
including the header.

## Version & format macros

```c
#define MLP_VERSION_MAJOR 0
#define MLP_VERSION_MINOR 9
#define MLP_VERSION_PATCH 1
#define MLP_VERSION_STRING "0.9.1"

#define MLP_MAGIC   0x4D4C5033u /* "MLP3" */
#define MLP_VERSION 3u
```

`MLP_VERSION_*`/`MLP_VERSION_STRING` describe the library release (see
[Versioning](../README.md#versioning) in the README). `MLP_MAGIC` and
`MLP_VERSION` are separate — they identify the on-disk **model file
format** used by `MLP_Save_Network`/`MLP_Load_Network`, and only change when that binary
format changes, independently of the library's own version number.
`MLP_VERSION` was bumped to `3` in `0.8.0` to serialize all metadata fields (`n_layers`,
`loss`, `neurons`, `inputs`, `activation`) as fixed-width `uint32_t` values, ensuring saved
model files are fully platform-independent and can be shared between 32-bit and 64-bit architectures.
Legacy model files (`MLP_VERSION 2` or `1`) will fail to load with `MLP_ERR_FILE_FORMAT` —
retrain and re-save them.

## Configuration macros

These may be `#define`d before including `MLP.h` to change library
behavior. Define them consistently across every translation unit that
includes the header.

```c
#define MLP_EXIT_ON_ERROR   // opt-in: abort on any public API failure
#define MLP_CSV_LINE_BUFFER 1024  // size of the internal CSV line buffer
#define MLP_USE_LIBM        // opt-in: use math.h functions instead of custom ones
#define MLPDEF              // customize API function linkage (defaults to extern)
```

- **`MLP_EXIT_ON_ERROR`** — if defined, every public API function that
  would otherwise set an error code and return `false`/a zeroed struct
  instead prints `MLP_ErrorString()` for the failure to `stderr` and
  calls `exit(EXIT_FAILURE)`. Useful for small programs/examples that
  don't want to check every return value; leave it undefined for
  libraries or applications that need to recover from errors.
- **`MLP_CSV_LINE_BUFFER`** — size in bytes of the stack buffer
  `MLP_LoadCSV()` uses to read one line at a time. Defaults to `1024`.
  A row longer than this (including its newline) fails with
  `MLP_ERR_CSV_LINE_TOO_LONG`; raise this value if your CSV has very
  wide rows. Must be defined (if at all) before including `MLP.h`.
- **`MLP_USE_LIBM`** — if defined, forces the library to use standard math
  functions from `<math.h>` (e.g. `sqrt`, `exp`, `log`) for weight initialization,
  activations, and loss calculations, rather than using the custom internal
  approximations. Requires linking against the standard math library (e.g. `-lm` on Unix).
  *Note: The library will also automatically detect and use `<math.h>` if the header has already been included in the translation unit before `MLP.h`.*
- **`MLPDEF`** — linkage macro used for all public API function declarations and
  definitions. Defaults to empty (external linkage, i.e., `extern`). Can be redefined
  (e.g., to `static inline` or custom visibility qualifiers) before including `MLP.h` to
  change function link scope, build a shared library, or configure the header for dead-code
  elimination when compiled header-only.

## Types

### `MLP_Error`

```c
typedef enum {
    MLP_OK = 0,

    MLP_ERR_NULL_POINTER,        // A required pointer argument was NULL
    MLP_ERR_INVALID_ARGUMENT,    // An argument was structurally invalid (e.g. zero-sized)
    MLP_ERR_ALLOC_FAILED,        // A heap allocation failed
    MLP_ERR_SHAPE_MISMATCH,      // Network/Dataset dimensions are incompatible

    MLP_ERR_FILE_OPEN,           // Could not open a file for reading/writing
    MLP_ERR_FILE_READ,           // A read from a file failed or was truncated
    MLP_ERR_FILE_WRITE,          // A write to a file failed or was truncated
    MLP_ERR_FILE_FORMAT,         // File contents did not match the expected format (bad magic/version)

    MLP_ERR_CSV_EMPTY,           // CSV contains no data rows.
    MLP_ERR_CSV_INVALID_NUMBER,  // A field is not a valid floating-point number.
    MLP_ERR_CSV_COLUMN_COUNT,    // A row has an unexpected number of columns.
    MLP_ERR_CSV_LINE_TOO_LONG,   // A CSV line exceeded the internal buffer size.
    MLP_ERR_CSV_HEADER,          // Invalid or missing CSV header.

    MLP_ERR_COUNT                // Sentinel: number of error codes, not a real error
} MLP_Error;
```

Every public function that can fail sets a thread-unaware global "last
error" on failure, retrievable with `MLP_GetLastError()`. Functions that
return a struct by value (`Dataset`, `Network`) signal failure by
returning a zeroed struct — check that, then consult
`MLP_GetLastError()` for the reason.

The `MLP_ERR_CSV_*` codes are only ever set by `MLP_LoadCSV()`; see that
function's entry below for what triggers each one. If `MLP_EXIT_ON_ERROR`
is defined, none of this matters for control flow — the process exits
before the error code would need to be checked — but `MLP_GetLastError()`
still reflects the failure in the `stderr` message printed on the way out.

### `Activation`

```c
typedef enum {
    ACT_LINEAR,
    ACT_RELU,
    ACT_LEAKY_RELU,
    ACT_SIGMOID,
    ACT_TANH,
    ACT_SOFTMAX,

    ACT_COUNT
} Activation;
```

Per-layer activation function, set individually for every layer via
`NetworkConfig.activations` (see `MLP_Create_Network` below).
`ACT_LINEAR` applies no nonlinearity (`f(z) = z`) and is the usual choice
for a regression output; `ACT_SIGMOID` squashes to `(0, 1)` and pairs
naturally with `LOSS_BINARY_CROSS_ENTROPY` for binary classification.
`ACT_TANH` applies hyperbolic tangent (`f(z) = tanh(z)`), squashing output
values to `(-1, 1)` and providing a zero-symmetric response.
`ACT_SOFTMAX` normalizes output values across all output neurons such that they
sum to `1.0`, expressing a probability distribution. It is only supported on the
network's final output layer (not hidden layers) and must be paired with
`LOSS_CATEGORICAL_CROSS_ENTROPY`.
`ACT_COUNT` is a sentinel, not a real activation.

### `Loss`

```c
typedef enum {
    LOSS_AUTO,

    LOSS_MSE,
    LOSS_BINARY_CROSS_ENTROPY,
    LOSS_CATEGORICAL_CROSS_ENTROPY,

    LOSS_COUNT
} Loss;
```

Set once per network via `NetworkConfig.loss`.
`LOSS_AUTO` is the default enum value (0) and is automatically selected when `.loss` is omitted or zero-initialized in `NetworkConfig`. It infers `LOSS_BINARY_CROSS_ENTROPY` for `ACT_SIGMOID` outputs, `LOSS_CATEGORICAL_CROSS_ENTROPY` for `ACT_SOFTMAX` outputs, and `LOSS_MSE` for all other output activations (`ACT_LINEAR`, `ACT_RELU`, `ACT_LEAKY_RELU`, `ACT_TANH`).
`LOSS_MSE` (mean squared error) suits regression and works with any output activation.
`LOSS_BINARY_CROSS_ENTROPY` is intended for a single sigmoid output representing a probability.
`LOSS_CATEGORICAL_CROSS_ENTROPY` is intended for multi-class classification (with output dimensions > 1) paired with an `ACT_SOFTMAX` output layer.
`LOSS_COUNT` is a sentinel, not a real loss.

### `Initializer`

```c
typedef enum {
    INIT_XAVIER,
    INIT_HE,

    INIT_COUNT
} Initializer;
```

Weight initialization strategy, set for each connection layer via `NetworkConfig.initializers`.
- `INIT_XAVIER` — Xavier/Glorot initialization, scaling initial uniform weights by `sqrt(1 / inputs)`. Best suited for linear or sigmoid activations.
- `INIT_HE` — He/Kaiming initialization, scaling initial uniform weights by `sqrt(2 / inputs)`. Best suited for ReLU or leaky ReLU activations.
- `INIT_COUNT` — sentinel, not a real initializer strategy.

### `TrainOptions`

```c
typedef struct {
    size_t      max_epochs;
    double      learning_rate;
    double      stop_loss;
    bool        verbose;
    const char *loss_file;
} TrainOptions;
```

| Field           | Meaning                                                        |
|-----------------|------------------------------------------------------------------|
| `max_epochs`    | Upper bound on training epochs.                                  |
| `learning_rate` | Step size used in gradient descent.                              |
| `stop_loss`     | Training stops early once mean epoch loss drops to or below this.|
| `verbose`       | If true, prints a progress bar and a summary line per epoch.     |
| `loss_file`     | File path to log the epoch loss history as CSV (disabled if `NULL`). |

Get sane defaults with `MLP_DefaultTrainOptions()`.

### `Dataset`

```c
typedef struct {
    double *inputs;  // Flattened: n_samples x n_features
    double *outputs; // Flattened: n_samples x n_outputs (may be NULL)

    size_t n_samples;
    size_t n_features;
    size_t n_outputs;
} Dataset;
```

A `Dataset` built with `MLP_Create_Dataset` does not own or copy
`inputs`/`outputs` — it just holds pointers into memory you manage, and
you're responsible for freeing that memory yourself. `outputs` may be
`NULL` for prediction-only datasets. Prefer constructing with
`MLP_Create_Dataset` rather than initializing the struct directly, since
it validates its inputs.

The exception is `MLP_LoadCSV()`: it heap-allocates `inputs`/`outputs`
itself, and the resulting `Dataset` **must** be freed with
`MLP_Destroy_Dataset()` rather than `free()`d or manually managed — don't
mix the two allocation styles for the same `Dataset`.

### `NetworkConfig`

```c
typedef struct {
    const size_t *topology;
    const size_t topology_size;

    const Activation *activations;
    const Initializer *initializers;

    Loss loss;
} NetworkConfig;
```

The input to `MLP_Create_Network`. `topology` lists unit counts from
input to output, e.g. `{2, 8, 1}` for a 2-input, 8-hidden, 1-output
network; `topology_size` is the length of that array (layer count + 1).
`activations` has one entry per *connection* — `topology_size - 1`
entries, i.e. one `Activation` per resulting `Layer` — so a 3-entry
topology needs a 2-entry `activations` array. `initializers` points to an
array of type `Initializer` of size `topology_size - 1` specifying the weight
initialization strategy for each connection layer. Pass `MLP_AUTO_INITIALIZERS`
(or `NULL`) to automatically assign the best initializer for each layer based
on its activation (He for ReLU/Leaky ReLU, Xavier for Sigmoid/Linear). `loss`
selects the training loss for the whole network (see `Loss` above).

### `Layer` / `Network`

```c
typedef struct {
    double *weights; // Flattened: neurons * inputs
    double *biases;  // neurons

    size_t neurons;
    size_t inputs;

    Activation activation;
} Layer;

typedef struct {
    Layer *layers;
    size_t n_layers;

    Loss loss;
} Network;
```

Exposed mainly so `MLP_View_Network` and custom inspection code can walk
the parameters directly. `weights` is a row-major `[neurons x inputs]`
matrix: `weights[j * inputs + k]` is the weight from input `k` to neuron
`j`. Each `Layer` carries its own `activation`, applied to that layer's
output during the forward pass; `Network.loss` is the loss it was built
with, used by `MLP_Train`.

## Functions

### `MLP_Create_Dataset`

```c
Dataset MLP_Create_Dataset(
    double *inputs,
    double *outputs,
    size_t n_samples,
    size_t n_features,
    size_t n_outputs
);
```

Builds a `Dataset` from existing arrays. Returns a zeroed `Dataset` (all
fields NULL/0) if `inputs` is NULL or `n_samples`/`n_features`/`n_outputs` is 0.

### `MLP_DefaultTrainOptions`

```c
TrainOptions MLP_DefaultTrainOptions(void);
```

Returns `{ max_epochs: 1000, learning_rate: 1e-3, stop_loss: 1e-8, verbose: false, loss_file: NULL }`.

### `MLP_Create_Network`

```c
Network MLP_Create_Network(const NetworkConfig *cfg);
```

> **Changed in 0.4.0:** this used to take a raw `(const size_t *topology,
> size_t n_layers)` pair. It now takes a single `NetworkConfig`, which
> adds per-layer activations and a selectable loss instead of a hardcoded
> leaky-ReLU-hidden/linear-output/MSE network. See the
> [0.4.0 changelog entry](../CHANGELOG.md) for the full rationale.

Builds a `Network` from a `NetworkConfig` (see above). Requires
`cfg`/`cfg->topology`/`cfg->activations` to be non-NULL,
`cfg->topology_size >= 2`, every topology entry `> 0`, every
`cfg->activations` entry `< ACT_COUNT`, `cfg->loss < LOSS_COUNT`, and every
`cfg->initializers` entry `< INIT_COUNT` (if `cfg->initializers` is non-NULL).
Weights are initialized according to the strategies configured in `cfg->initializers`
(or automatically selected if `cfg->initializers` is `MLP_AUTO_INITIALIZERS`/`NULL`;
see `Initializer` above); biases start at 0. Returns a zeroed `Network` on
invalid input or allocation failure.

Weight randomization (which forms the base or scale for the initializers)
uses `rand()`, uninitialized by the library — call `srand()` yourself if
you want reproducible or non-deterministic runs.

```c
Network net = MLP_Create_Network(&(NetworkConfig){
    .topology      = (size_t[]){ 2, 8, 1 },
    .topology_size = 3,
    .activations   = (Activation[]){ ACT_LEAKY_RELU, ACT_LINEAR },
    .initializers  = MLP_AUTO_INITIALIZERS,
    .loss          = LOSS_MSE,
});
```

### `MLP_View_Network` / `MLP_View_Dataset`

```c
void MLP_View_Network(const Network *net);
void MLP_View_Dataset(const Dataset *d);
```

Debug helpers that print weights/biases or a tabular data dump to
`stdout`. No-ops on NULL/empty input.

### `MLP_Train`

```c
bool MLP_Train(Network *net, const Dataset *d, TrainOptions *options);
```

Trains `net` in place via per-sample SGD backpropagation, minimizing
whichever `Loss` the network was created with (`net->loss` — see
`NetworkConfig`). Any zeroed fields in `options` (e.g. `max_epochs == 0`, `learning_rate == 0.0`, or `stop_loss == 0.0`) are automatically populated with defaults from `MLP_DefaultTrainOptions()`. Returns `false` if `net`, `d`, or `options` are
invalid, if `d->output` is NULL, or if the dataset's feature/output
counts don't match the network's input/output widths. Returns `true` on
completion (including early stop via `stop_loss`). Both `stop_loss` and the
printed loss are computed using the network's configured `Loss` function (i.e.,
mean squared error for `LOSS_MSE`, binary cross-entropy for
`LOSS_BINARY_CROSS_ENTROPY`, or categorical cross-entropy for `LOSS_CATEGORICAL_CROSS_ENTROPY`), normalized by the total number of prediction
outputs (`d->n_samples * d->n_outputs`).

### `MLP_Predict`

```c
bool MLP_Predict(const Network *net, double *input, double *output);
```

Runs a forward pass on a single input vector `input` and writes the predicted output to `output`.
- `input` must point to an array of size equal to the network's input features count (`net->layers[0].inputs`).
- `output` must point to a caller-allocated array of size equal to the network's output count (`net->layers[net->n_layers - 1].neurons`).

Returns `false` if `net`, `input`, or `output` is NULL. Otherwise returns `true` on success.

### `MLP_Predict_Dataset`

```c
bool MLP_Predict_Dataset(const Network *net, const Dataset *d, double *buf);
```

Runs a forward pass for every sample in `d` and writes outputs into `buf`
(caller-allocated, sized `d->n_samples * d->n_outputs`). `d->output` may
be NULL. Returns `false` on shape mismatch or invalid input.

### `MLP_Destroy_Network`

```c
void MLP_Destroy_Network(Network *net);
```

Frees all layer weights/biases and the layer array, then zeroes `net`.
Safe to call on an already-destroyed or zero-initialized `Network`.

### `MLP_Save_Network`

```c
bool MLP_Save_Network(const Network *net, const char *filename);
```

Writes `net` to `filename` in the library's binary model format (magic
number `MLP_MAGIC`, format version `MLP_VERSION`, followed by
`n_layers`, the network's `loss`, and each layer's shape, activation,
weights, and biases). Returns `false`
if `net`/`net->layers`/`filename` is NULL, or if any file write fails —
on write failure the partially-written file is deleted. Overwrites
`filename` if it already exists.

### `MLP_Load_Network`

```c
bool MLP_Load_Network(Network *net, const char *filename);
```

Reads a model previously written by `MLP_Save_Network()` from `filename` and
populates `net`, allocating the layer array and each layer's
weights/biases. `net` may point to a zero-initialized `Network` or an
existing one — either way, any network `*net` currently holds is
destroyed (via `MLP_Destroy_Network`) before the new one is loaded.
Returns `false` if `net`/`filename` is NULL, if the file can't be
opened, if the magic number or version doesn't match (`MLP_MAGIC`,
`MLP_VERSION` — i.e. the file isn't a valid MLP.h model or was written
by an incompatible version), or if any read is short/fails — on failure
`*net` is left zeroed rather than partially populated.

### `MLP_LoadCSV`

```c
Dataset MLP_LoadCSV(
    const char *filename,
    size_t max_samples,
    size_t n_features,
    size_t n_outputs,
    bool has_header
);
```

Reads up to `max_samples` rows from a comma-separated `filename` into a
newly heap-allocated `Dataset`. Each row must have exactly
`n_features + n_outputs` numeric columns: the first `n_features` become
`samples`, the remaining `n_outputs` become `output`. If `n_outputs` is
`0`, `output` stays `NULL` (useful for prediction-only CSVs). Set
`has_header` to `true` to skip the first line.

Rows are read one at a time into a stack buffer sized
`MLP_CSV_LINE_BUFFER` (`1024` bytes by default, overridable — see
[Configuration macros](#configuration-macros)). If fewer than
`max_samples` rows are actually present, the backing arrays are
`realloc`'d down to the exact row count on success.

Returns a zeroed `Dataset` on failure, with `MLP_GetLastError()` set to
one of:

| Cause                                   | Error                          |
|------------------------------------------|--------------------------------|
| `filename` is `NULL`                     | `MLP_ERR_NULL_POINTER`         |
| `n_features` or `max_samples` is `0`, or the requested size overflows `size_t` | `MLP_ERR_INVALID_ARGUMENT` |
| File can't be opened                     | `MLP_ERR_FILE_OPEN`            |
| Allocation of `samples`/`output` fails   | `MLP_ERR_ALLOC_FAILED`         |
| `has_header` is true but the file has no first line | `MLP_ERR_CSV_HEADER` |
| A line (including its newline) exceeds `MLP_CSV_LINE_BUFFER` | `MLP_ERR_CSV_LINE_TOO_LONG` |
| A field doesn't parse as a valid `double` | `MLP_ERR_CSV_INVALID_NUMBER`  |
| A row has more or fewer than `n_features + n_outputs` columns | `MLP_ERR_CSV_COLUMN_COUNT` |
| A read error occurs partway through the file | `MLP_ERR_FILE_READ`        |
| The file has a header (or is otherwise consumed) but zero data rows | `MLP_ERR_CSV_EMPTY` |

A `Dataset` returned by `MLP_LoadCSV()` must be released with
`MLP_Destroy_Dataset()`, not `free()`.

### `MLP_Destroy_Dataset`

```c
void MLP_Destroy_Dataset(Dataset *d);
```

Frees `d->samples` and `d->output` (either may be `NULL`) and zeroes
`*d`. Use this to release a `Dataset` returned by `MLP_LoadCSV()`. Safe
to call on an already-destroyed or zero-initialized `Dataset`, and a
no-op if `d` itself is `NULL`.

Do **not** call this on a `Dataset` built with `MLP_Create_Dataset` —
that variant doesn't own its arrays, so `MLP_Destroy_Dataset` would free
memory you're still responsible for managing yourself.

### `MLP_GetLastError`

```c
MLP_Error MLP_GetLastError(void);
```

Returns the `MLP_Error` set by the most recently failed public API call.
Starts as `MLP_OK` and is only ever updated on failure — a successful
call does not reset it back to `MLP_OK`, so check function return values
first and treat this as "the reason for the last failure," not "the
current status."

### `MLP_ErrorString`

```c
const char *MLP_ErrorString(MLP_Error err);
```

Returns a static, human-readable description of `err` (e.g. `"Memory
allocation failed"`). Typically called as
`MLP_ErrorString(MLP_GetLastError())` right after a function returns
`false`/zeroed. Unknown values return `"Unknown error"`.

### `MLP_Perror`

```c
void MLP_Perror(const char *str);
```

Prints `str` followed by a human-readable description of the last error (equivalent to `MLP_ErrorString(MLP_GetLastError())`) to standard output. Designed to match standard library `perror` usage.

## Notes on the private API

Everything prefixed `_` (`_forward`, `_backprop`, `_Workspace_*`,
`_ReLU`, `_initialize_weight`, `_print_summary`) is implementation detail. It
is declared `static` and not part of the stable API — signatures may
change between versions without a deprecation notice.