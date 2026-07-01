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

`MLP_Create_Network` takes a topology array: the number of units in each
layer, from input to output.

```c
size_t topology[] = {2, 8, 1}; // 2 inputs -> 8 hidden -> 1 output
Network net = MLP_Create_Network(topology, 3);
```

Hidden layers use a leaky ReLU activation; the output layer is always
linear (no activation), which makes the library equally usable for
regression and classification-via-thresholding.

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

Predictions are written into a caller-provided buffer sized
`n_samples * n_outputs`.

```c
double preds[4];
Dataset test = MLP_Create_Dataset(X, NULL, 4, 2, 1);
MLP_Predict_Dataset(&net, &test, preds);
```

## 6. Clean up

```c
MLP_Destroy_Network(&net);
```

Datasets don't own their data (they just point at your arrays), so there's
nothing to free there.