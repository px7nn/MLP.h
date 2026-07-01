# API Reference

All public symbols are declared in `MLP.h`. Implementations are only
compiled in the translation unit that defines `MLP_IMPLEMENTATION` before
including the header.

## Types

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

## Notes on the private API

Everything prefixed `_` (`_forward`, `_backprop`, `_Workspace_*`,
`_ReLU`, `_random_weight`, `_print_summary`) is implementation detail. It
is declared `static` and not part of the stable API — signatures may
change between versions without a deprecation notice.
