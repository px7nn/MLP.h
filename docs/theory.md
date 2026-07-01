# Theory: How MLP.h Works

A short, implementation-focused explanation of the math behind
`_forward`, `_backprop`, and the training loop. This isn't a general
neural-network tutorial — it's a map from the equations to the exact
lines of code that implement them.

## Architecture

A `Network` is a sequence of `Layer`s. Each layer fully connects its
`inputs` to its `neurons`. For a topology `{2, 8, 1}`, `MLP_Create_Network`
builds two `Layer`s: one mapping 2 -> 8, one mapping 8 -> 1.

- **Hidden layers** use leaky ReLU: `f(z) = z if z > 0 else 0.01*z`
  (`_ReLU` in the source).
- **The output layer is always linear** — no activation applied. This
  keeps the library usable for both regression (real-valued targets) and
  classification (threshold or argmax the raw outputs yourself).

## Forward pass (`_forward`)

For layer `i`, with weight matrix `W` (shape `neurons x inputs`), bias
vector `b`, and input activation vector `a`:

```
z[j] = b[j] + sum_k W[j][k] * a[k]
a'[j] = ReLU(z[j])          (hidden layers)
a'[j] = z[j]                (output layer)
```

The workspace (`_Workspace`) stores one activation vector per layer,
including a slot for the raw input (`activations[0]`), so
`activations[i+1]` is always the output of `net->layers[i]`.

## Loss

Mean squared error over each sample's outputs:

```
L = (1/n_outputs) * sum_i (pred[i] - target[i])^2
```

accumulated per-sample and averaged across the epoch in `MLP_Train`.

## Backward pass (`_backprop`)

Backprop computes a "delta" (error signal) per neuron, then that delta
plus the corresponding upstream activation gives the gradient for each
weight.

**Output layer delta** is just the residual (since MSE's derivative w.r.t.
a linear output simplifies to the plain error):

```
delta_output[i] = pred[i] - target[i]
```

**Hidden layer delta** propagates the next layer's deltas backward through
that layer's weights, then multiplies by the local activation derivative:

```
delta[i][j] = ReLU'(a[i][j]) * sum_k ( delta[i+1][k] * W_next[k][j] )
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

## Why leaky ReLU?

Plain ReLU can "die" (a neuron whose pre-activation is always negative
stops learning entirely, since its gradient is always zero). The leaky
variant (`0.01 * z` for `z <= 0`) keeps a small gradient flowing even in
the negative region, which is a cheap way to avoid that failure mode
without adding a hyperparameter to tune.

## Known limitations

- No batching — every sample triggers an immediate weight update.
- No regularization (L1/L2, dropout) or momentum-based optimizers.
- No softmax/cross-entropy output stage — multi-class classification has to be done via one-hot MSE regression and
  argmax, which works but isn't the theoretically ideal loss for
  classification.
- Weight initialization is uniform in `[-1, 1]`, not scaled by fan-in/out
  (e.g. no Xavier/He initialization), so very deep or wide networks may
  be harder to train.

These are reasonable directions for future versions.
