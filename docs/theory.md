# Theory: How MLP.h Works

A short, implementation-focused explanation of the math behind
`_forward`, `_backprop`, and the training loop. This isn't a general
neural-network tutorial — it's a map from the equations to the exact
lines of code that implement them.

## Architecture

A `Network` is a sequence of `Layer`s. Each layer fully connects its
`inputs` to its `neurons`. For a topology `{2, 8, 1}`, `MLP_Create_Network`
builds two `Layer`s: one mapping 2 -> 8, one mapping 8 -> 1.

Each layer's activation function is configured individually via
`NetworkConfig.activations` (one entry per layer, i.e. `topology_size - 1`
entries). Four are available:

- `ACT_LINEAR` — `f(z) = z`. Typical choice for a regression output.
- `ACT_RELU` — `f(z) = max(0, z)`.
- `ACT_LEAKY_RELU` — `f(z) = z if z > 0 else 0.01*z`. A common default
  for hidden layers; keeps a small gradient flowing when `z <= 0`, which
  avoids plain ReLU's "dead neuron" failure mode without adding a
  hyperparameter to tune.
- `ACT_SIGMOID` — squashes to `(0, 1)`; pairs naturally with
  `LOSS_BINARY_CROSS_ENTROPY` for binary classification output.

There's no architectural restriction tying a particular activation to
hidden vs. output layers — any layer can use any of the four.

## Weight initialization

Initial weights are set by `_initialize_weight` based on the `Initializer` strategy configured per layer in `NetworkConfig.initializers`. If `cfg->initializers` is set to `MLP_AUTO_INITIALIZERS` (or `NULL`), the library automatically selects the best strategy for each layer based on its activation function:

- **`INIT_XAVIER`**: Xavier/Glorot initialization, scaling initial uniform weights by `sqrt(1 / inputs)`:  
  `w[j][k] = sqrt(1 / inputs) * r` where `r ~ U(-1, 1)`.  
  Useful for keeping signal variance stable across layers. Automatically selected for `ACT_SIGMOID` and `ACT_LINEAR`.
- **`INIT_HE`**: He/Kaiming initialization, scaling initial uniform weights by `sqrt(2 / inputs)`:  
  `w[j][k] = sqrt(2 / inputs) * r` where `r ~ U(-1, 1)`.  
  Recommended to account for zeroed-out negative activations. Automatically selected for `ACT_RELU` and `ACT_LEAKY_RELU`.

Biases for all layers are initialized to `0` regardless of the weight initializer.

## Forward pass (`_forward`)

For layer `i`, with weight matrix `W` (shape `neurons x inputs`), bias
vector `b`, and input activation vector `a`:

```
z[j] = b[j] + sum_k W[j][k] * a[k]
a'[j] = f(z[j])          (f = that layer's configured Activation)
```

The workspace (`_Workspace`) stores one activation vector per layer,
including a slot for the raw input (`activations[0]`), so
`activations[i+1]` is always the output of `net->layers[i]`.

## Loss

Set once per network via `NetworkConfig.loss`:

- **`LOSS_MSE`** — mean squared error, suits regression and works with
  any output activation:

```
L = (1/n_outputs) * sum_i (pred[i] - target[i])^2
```

- **`LOSS_BINARY_CROSS_ENTROPY`** — intended for a single `ACT_SIGMOID`
  output representing a probability:

```
L = -(1/n_outputs) * sum_i [ target[i]*log(pred[i]) + (1-target[i])*log(1-pred[i]) ]
```

Note that the reported loss and early-stop checks in `MLP_Train` are calculated
using the active loss function configured for the network, rather than defaulting
to mean squared error. Average loss values are normalized across both samples and
output units (`n_samples * n_outputs`).

## Backward pass (`_backprop`)

Backprop computes a "delta" (error signal) per neuron, then that delta
plus the corresponding upstream activation gives the gradient for each
weight.

**Output layer delta** depends on `net->loss`:

```
LOSS_MSE:                   delta_output[i] = (pred[i] - target[i]) * f'(pred[i])
LOSS_BINARY_CROSS_ENTROPY:  delta_output[i] = (pred[i] - target[i])
```

where `f'` is the output layer's activation derivative. For
`LOSS_MSE` this is the plain chain-rule derivative. For
`LOSS_BINARY_CROSS_ENTROPY` the `f'(pred[i])` term cancels out of the true
derivative when paired with a sigmoid output, leaving the simplified
`pred - target` form — this is what `MLP_Train` computes regardless of
what activation the output layer actually uses, so pairing
`LOSS_BINARY_CROSS_ENTROPY` with a non-sigmoid output layer will still
train, but the gradient won't correspond to that loss's true derivative.

**Hidden layer delta** propagates the next layer's deltas backward through
that layer's weights, then multiplies by the local layer's own
activation derivative:

```
delta[i][j] = f_i'(a[i][j]) * sum_k ( delta[i+1][k] * W_next[k][j] )
```

The `_backprop` loop walks layers from output back to the first hidden
layer (skipping the input layer, which has no learnable parameters and
thus no delta to compute). Note the weight indexing here is "transposed"
relative to the forward pass — forward reads `W[k][j]` per output neuron
`k`'s row, backward reads down column `j` across all `k`, since error
is flowing in the opposite direction from activation.

## Gradient descent step

Once every layer's deltas are known, weight and bias gradients follow
directly from the chain rule:

```
dL/dW[j][k] = delta[j] * a_in[k]
dL/db[j]    = delta[j]
```

`MLP_Train` applies these directly (plain SGD, no momentum or
regularization):

```
W[j][k] -= learning_rate * delta[j] * a_in[k]
b[j]    -= learning_rate * delta[j]
```

Updates happen **per sample**, not once per epoch (this is stochastic
gradient descent, not batch gradient descent) — which is simple to reason
about but means training speed and stability are sensitive to
`learning_rate` and dataset ordering.

## Known limitations

- No batching — every sample triggers an immediate weight update.
- No regularization (L1/L2, dropout) or momentum-based optimizers.
- No softmax/cross-entropy output stage for multi-class problems — only
  binary cross-entropy (single sigmoid output) is supported; multi-class
  classification has to be done via one-hot MSE regression and argmax,
  which works but isn't the theoretically ideal loss for that case.

These are reasonable directions for future versions.