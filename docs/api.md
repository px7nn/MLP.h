# API Reference

All public symbols are declared in `MLP.h`. Implementations are only
compiled in the translation unit that defines `MLP_IMPLEMENTATION` before
including the header.

## Version & format macros

```c
#define MLP_VERSION_MAJOR 0
#define MLP_VERSION_MINOR 2
#define MLP_VERSION_PATCH 0
#define MLP_VERSION_STRING "0.2.0"

#define MLP_MAGIC   0x4D4C5031u /* "MLP1" */
#define MLP_VERSION 1u
```

`MLP_VERSION_*`/`MLP_VERSION_STRING` describe the library release (see
[Versioning](../README.md#versioning) in the README). `MLP_MAGIC` and
`MLP_VERSION` are separate — they identify the on-disk **model file
format** used by `MLP_Save_Network`/`MLP_Load_Network`, and only change when that binary
format changes, independently of the library's own version number.

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

    MLP_ERR_COUNT                // Sentinel: number of error codes, not a real error
} MLP_Error;
```

Every public function that can fail sets a thread-unaware global "last
error" on failure, retrievable with `MLP_GetLastError()`. Functions that
return a struct by value (`Dataset`, `Network`) signal failure by
returning a zeroed struct — check that, then consult
`MLP_GetLastError()` for the reason.

### `TrainOptions`

```c
typedef struct {
    size_t max_epochs;
    double learning_rate;
    double stop_loss;
    bool   verbose;
} TrainOptions;
```

| Field           | Meaning                                                        |
|-----------------|------------------------------------------------------------------|
| `max_epochs`    | Upper bound on training epochs.                                  |
| `learning_rate` | Step size used in gradient descent.                              |
| `stop_loss`     | Training stops early once mean epoch loss drops to or below this.|
| `verbose`       | If true, prints a progress bar and a summary line per epoch.     |

Get sane defaults with `MLP_DefaultTrainOptions()`.

### `Dataset`

```c
typedef struct {
    const double *samples; // Flattened: samples x features
    double *output;        // Flattened: samples x n_outputs (may be NULL)

    const size_t n_samples;
    const size_t n_features;
    const size_t n_outputs;
} Dataset;
```

A `Dataset` does not own or copy `samples`/`output` — it just holds
pointers into memory you manage. `output` may be `NULL` for
prediction-only datasets. Construct with `MLP_Create_Dataset`, never
directly (the `const` shape fields make direct initialization awkward and
`MLP_Create_Dataset` validates inputs).

### `Layer` / `Network`

```c
typedef struct {
    double *weights; // Flattened: neurons * inputs
    double *biases;  // neurons
    size_t neurons;
    size_t inputs;
} Layer;

typedef struct {
    Layer *layers;
    size_t n_layers;
} Network;
```

Exposed mainly so `MLP_View_Network` and custom inspection code can walk
the parameters directly. `weights` is a row-major `[neurons x inputs]`
matrix: `weights[j * inputs + k]` is the weight from input `k` to neuron
`j`.

## Functions

### `MLP_Create_Dataset`

```c
Dataset MLP_Create_Dataset(
    const double *samples,
    double *output,
    size_t n_samples,
    size_t n_features,
    size_t n_outputs
);
```

Builds a `Dataset` from existing arrays. Returns a zeroed `Dataset` (all
fields NULL/0) if `samples` is NULL or `n_samples`/`n_features` is 0.

### `MLP_DefaultTrainOptions`

```c
TrainOptions MLP_DefaultTrainOptions(void);
```

Returns `{ max_epochs: 1000, learning_rate: 1e-3, stop_loss: 1e-8, verbose: false }`.

### `MLP_Create_Network`

```c
Network MLP_Create_Network(const size_t *topology, size_t n_layers);
```

`topology` lists unit counts from input to output, e.g. `{2, 8, 1}` for a
2-input, 8-hidden, 1-output network. `n_layers` is the length of
`topology` (despite the name, it's the topology array length, i.e.
layer-count + 1). Requires `n_layers >= 2` and every topology entry > 0.
Weights are randomly initialized in `[-1, 1]`; biases start at 0. Returns
a zeroed `Network` on invalid input or allocation failure.

Weight randomization uses `rand()`, uninitialized by the library — call
`srand()` yourself if you want reproducible or non-deterministic runs.

### `MLP_View_Network` / `MLP_View_Dataset`

```c
void MLP_View_Network(const Network *net);
void MLP_View_Dataset(const Dataset *d);
```

Debug helpers that print weights/biases or a tabular data dump to
`stdout`. No-ops on NULL/empty input.

### `MLP_Train`

```c
bool MLP_Train(Network *net, const Dataset *d, const TrainOptions *options);
```

Trains `net` in place via per-sample SGD backpropagation, minimizing mean
squared error. Returns `false` if `net`, `d`, or `options` are invalid, if
`d->output` is NULL, or if the dataset's feature/output counts don't
match the network's input/output widths. Returns `true` on completion
(including early stop via `stop_loss`).

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
`n_layers` and each layer's shape, weights, and biases). Returns `false`
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

## Notes on the private API

Everything prefixed `_` (`_forward`, `_backprop`, `_Workspace_*`,
`_ReLU`, `_random_weight`, `_print_summary`) is implementation detail. It
is declared `static` and not part of the stable API — signatures may
change between versions without a deprecation notice.