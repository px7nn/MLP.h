# Getting Started

MLP.h is a single-header C library. There's nothing to build or link — you
just drop `MLP.h` into your project.

## 1. Include it

In exactly **one** `.c` file, define `MLP_IMPLEMENTATION` before including
the header. This pulls in the function bodies. Every other file that needs
the API can include `MLP.h` without the define.

```c
#define MLP_IMPLEMENTATION
#include "MLP.h"
```

## 2. Build a network

`MLP_Create_Network` takes a `NetworkConfig`: a topology array (unit
count per layer, from input to output), one `Activation` per layer, and
a `Loss` to train against.

```c
NetworkConfig cfg = {
    .topology      = (size_t[]){2, 8, 1}, // 2 inputs -> 8 hidden -> 1 output
    .topology_size = 3,
    .activations   = (Activation[]){ACT_LEAKY_RELU, ACT_LINEAR},
    .initializers  = MLP_AUTO_INITIALIZERS, // or custom array, e.g. (Initializer[]){INIT_HE, INIT_XAVIER}
    .loss          = LOSS_MSE,
};

Network net = MLP_Create_Network(&cfg);
```

`activations` has one entry per *connection*, i.e. `topology_size - 1`
entries — here that's "input->hidden" (`ACT_LEAKY_RELU`) and
"hidden->output" (`ACT_LINEAR`). A linear, unactivated output works for
both regression and classification-via-thresholding; for a probability
output, use `ACT_SIGMOID` on the last layer paired with
`LOSS_BINARY_CROSS_ENTROPY` (see [`examples/load_csv.c`](../examples/load_csv.c)).
`initializers` configures the weight initialization strategy for each layer
(connection). Pass `MLP_AUTO_INITIALIZERS` (or `NULL`) to let the library
automatically assign the best initializer for each layer based on its activation
(He initialization for ReLU/Leaky ReLU, and Xavier for Sigmoid/Linear/Tanh). Or,
pass a custom array of size `topology_size - 1` (e.g. `(Initializer[]){INIT_HE, INIT_XAVIER}`).

## 3. Wrap your data in a Dataset

Data is passed as flat, row-major arrays — no structs-of-arrays, no
copying. `MLP_Create_Dataset` just bundles pointers and shape info.

```c
double X[8] = {0,0, 0,1, 1,0, 1,1}; // 4 samples x 2 features
double Y[4] = {0, 1, 1, 0};         // 4 samples x 1 output

Dataset d = MLP_Create_Dataset(X, Y, 4, 2, 1);
```

`output` may be `NULL` for a dataset you only intend to predict on (no
labels needed for inference).

### Loading data from a CSV instead

If your data already lives in a file, `MLP_LoadCSV` reads it directly
into a `Dataset` — no manual array wrangling required. Each row must
have exactly `n_features + n_outputs` numeric columns, features first:

```c
Dataset d = MLP_LoadCSV(
    "circle.csv",
    /* max_samples */ 30,
    /* n_features  */ 2,
    /* n_outputs   */ 1,
    /* has_header  */ true
);
```

Unlike `MLP_Create_Dataset`, this allocates its own `samples`/`output`
arrays on the heap — free them with `MLP_Destroy_Dataset(&d)` (not
`free()`) once you're done. See
[`examples/load_csv.c`](../examples/load_csv.c) for a full example.

## 4. Train

```c
TrainOptions opt = MLP_DefaultTrainOptions();
opt.max_epochs    = 5000;
opt.learning_rate = 0.05;
opt.verbose       = true; // prints a progress bar + loss

MLP_Train(&net, &d, &opt);
```

`MLP_Train` runs full-batch-per-sample SGD (it updates weights after every
sample, not once per epoch) and minimizes mean squared error.

## 5. Predict

You can predict on an entire dataset at once or run inference on a single sample:

### Batch Prediction

Predictions for a whole dataset are written into a caller-provided buffer sized `n_samples * n_outputs`.

```c
double preds[4];
Dataset test = MLP_Create_Dataset(X, NULL, 4, 2, 1);
MLP_Predict_Dataset(&net, &test, preds);
```

### Single-sample Prediction

To predict on a single input array directly, use `MLP_Predict()`:

```c
double input[2] = {0, 1};
double output[1];
MLP_Predict(&net, input, output);
```

## 6. Save and load a model

Once trained, a network can be written to disk and loaded back later
without retraining:

```c
if (!MLP_Save_Network(&net, "xor.mlp")) {
    printf("Save failed: %s\n", MLP_ErrorString(MLP_GetLastError()));
}
```

```c
Network loaded = {0};

if (!MLP_Load_Network(&loaded, "xor.mlp")) {
    printf("Load failed: %s\n", MLP_ErrorString(MLP_GetLastError()));
}
```

`MLP_Load_Network` allocates everything it needs, so `loaded` just needs to be
zero-initialized (or a network you're fine seeing destroyed and
replaced) before the call.

## 7. Handle errors

Every public function that can fail sets a retrievable error code
instead of (or in addition to) returning `false`/a zeroed struct:

```c
Network net = MLP_Create_Network(NULL);
if (!net.layers) {
    printf("%s\n", MLP_ErrorString(MLP_GetLastError()));
    // "A required argument was NULL"
}
```

Call `MLP_ErrorString(MLP_GetLastError())` right after a failing call to
get a human-readable reason.

## 8. Clean up

```c
MLP_Destroy_Network(&net);
```

A `Dataset` built with `MLP_Create_Dataset` doesn't own its data (it just
points at your arrays), so there's nothing to free there. A `Dataset`
returned by `MLP_LoadCSV`, on the other hand, *does* own heap-allocated
arrays and must be released with `MLP_Destroy_Dataset`:

```c
MLP_Destroy_Dataset(&d);
```

## 9. Fail fast (optional)

For small scripts where you'd rather not check every return value,
define `MLP_EXIT_ON_ERROR` before including the header:

```c
#define MLP_EXIT_ON_ERROR
#define MLP_IMPLEMENTATION
#include "MLP.h"
```

With this defined, any public API call that would normally return
`false`/a zeroed struct instead prints the error and calls
`exit(EXIT_FAILURE)` immediately. Leave it undefined in code that needs
to recover from errors instead of aborting.

## Full examples

The [`examples/`](../examples/) directory has four runnable programs:

- [`xor_gate.c`](../examples/xor_gate.c) — trains a network on XOR from
  scratch and saves it to `xor.mlp`.
- [`load_model.c`](../examples/load_model.c) — loads `xor.mlp` (run
  `xor_gate.c` first) and runs inference without retraining.
- [`load_csv.c`](../examples/load_csv.c) — loads
  [`datasets/circle.csv`](../datasets/circle.csv) with `MLP_LoadCSV`,
  trains on it, predicts, and cleans up with `MLP_Destroy_Dataset`.
- [`visual_sin.c`](../examples/visual_sin.c) — loads
  [`datasets/sinewave_train.csv`](../datasets/sinewave_train.csv), trains a
  network to fit a sine wave, predicts outputs over interpolation and
  extrapolation ranges, and plots the results using gnuplot.