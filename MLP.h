/*=============================================================================
    MLP.h — A tiny single-header Multi-Layer Perceptron library in C

    A minimal, dependency-free feedforward neural network implementation
    supporting arbitrary topologies, ReLU hidden activations, linear output,
    and gradient-descent training via backpropagation on squared error loss.

    Usage: define MLP_IMPLEMENTATION in exactly one translation unit before
    including this header to pull in the implementation.

    Version: 0.1.0
    License: MIT (see LICENSE)
=============================================================================*/

#ifndef MLP_H
#define MLP_H

#define MLP_VERSION_MAJOR 0
#define MLP_VERSION_MINOR 1
#define MLP_VERSION_PATCH 0
#define MLP_VERSION_STRING "0.1.0"


/*=============================================================================
    Standard Library
=============================================================================*/

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>


/*=============================================================================
    Public Data Structures
=============================================================================*/

typedef struct {
    size_t max_epochs;
    double learning_rate;
    double stop_loss;
    bool verbose;
} TrainOptions;

typedef struct {
    const double *samples; // Flattened: samples x features
    double *output;        // Flattened: samples x n_outputs (may be NULL for prediction-only datasets)

    const size_t n_samples; 
    const size_t n_features;
    const size_t n_outputs;
} Dataset;

typedef struct {
    double *weights; // Flattened: neurons * inputs
    double *biases;  // neurons

    size_t neurons;  // number of neurons in this layer (its output width)
    size_t inputs;   // number of inputs into this layer (previous layer's neuron count)
} Layer;

typedef struct {
    Layer *layers;
    size_t n_layers;
} Network;


/*=============================================================================
    Private Data Structures
=============================================================================*/

typedef struct {
    double **activations; // layers * topology_size[i]
    double **deltas;      // layers * topology_size[i]

    size_t layers;
} _Workspace;


/*=============================================================================
    Public API
=============================================================================*/

static Dataset MLP_Create_Dataset(
    const double *samples, 
    double *output, 
    const size_t n_samples, 
    const size_t n_features,
    const size_t n_outputs
);

static TrainOptions MLP_DefaultTrainOptions(void);

static Network MLP_Create_Network(const size_t *topology, const size_t n_layers);
static void    MLP_View_Network(const Network *net);
static void    MLP_View_Dataset(const Dataset *d);
static bool    MLP_Train(Network *net, const Dataset *d, const TrainOptions *options);
static bool    MLP_Predict_Dataset(const Network *net, const Dataset *d, double *buf);
static void    MLP_Destroy_Network(Network *net);


/*=============================================================================
    Private API
=============================================================================*/


static void _print_summary(size_t epoch, size_t max_epochs, double loss, const char *reason);

static inline double _random_weight(void);
static inline double _ReLU(double z);
static inline double _ReLU_derivative(double activation);


static _Workspace _Workspace_Create(const Network *net);
static void       _Workspace_Destroy(_Workspace *ws);

static void _forward(
    _Workspace *ws, 
    const Network *net, 
    const double *sample
);

static void _backprop(
    _Workspace *ws, 
    const Network *net, 
    const double *target
);


/*=============================================================================
    Implementation
=============================================================================*/

#ifdef MLP_IMPLEMENTATION


static void _print_summary(size_t epoch, size_t max_epochs, double loss, const char *reason){
    if(!reason){
        const size_t width = 40;

        const size_t filled = (epoch * width) / max_epochs;
        printf("\r[");

        for(size_t i=0; i<width; ++i)
            putchar(i < filled ? '#':'-');

        printf("] %3zu%% Epoch %5zu/%5zu  Loss %.3e",
            (epoch * 100) / max_epochs,
            epoch,
            max_epochs,
            loss
        );

        fflush(stdout);
    } else {
        printf("\n\nTraining completed.\n\n");
        printf("Epochs     : %zu\n", epoch);
        printf("Final Loss : %.8e\n", loss);
        printf("Reason     : %s\n\n", reason);
    }
}


static inline double _random_weight(void){
    return (((double)rand() / RAND_MAX) * 2.0 - 1.0);
}

static inline double _ReLU(double z){
    return (z > 0.0)? z : 0.01 * z;
}

static inline double _ReLU_derivative(double activation){
    return (activation > 0.0)? 1.0 : 0.01;
}


static _Workspace _Workspace_Create(const Network *net){
    _Workspace ws = {0};

    if (!net || !net->layers || net->n_layers == 0)
        return ws;

    // +1 because activations/deltas include slot 0 for the raw input,
    // so activations[i+1]/deltas[i+1] line up with net->layers[i]'s output.
    ws.layers = net->n_layers + 1;

    ws.activations = calloc(ws.layers, sizeof *ws.activations);
    ws.deltas     = calloc(ws.layers, sizeof *ws.deltas);

    if(!ws.activations || !ws.deltas)
        goto fail;

    ws.activations[0] = calloc(net->layers[0].inputs, sizeof *ws.activations[0]);

    if(!ws.activations[0])
        goto fail;

    for(size_t i=0; i < net->n_layers; ++i){

        ws.activations[i+1] = calloc(
            net->layers[i].neurons, 
            sizeof *ws.activations[i+1]
        );

        ws.deltas[i+1] = calloc(
            net->layers[i].neurons, 
            sizeof *ws.deltas[i+1]
        );

        if(!ws.activations[i+1] || !ws.deltas[i+1])
            goto fail;
    }

    return ws;

    fail:
        if (ws.activations) {
            for (size_t i = 0; i < ws.layers; ++i)
                free(ws.activations[i]);
            free(ws.activations);
        }

        if (ws.deltas) {
            for (size_t i = 0; i < ws.layers; ++i)
                free(ws.deltas[i]);
            free(ws.deltas);
        }

        return (_Workspace){0};
}

static void _Workspace_Destroy(_Workspace *ws){
    if(!ws || !ws->activations)
        return;

    for(size_t i = 0; i < ws->layers; ++i){
        free(ws->activations[i]);
        free(ws->deltas[i]);
    }
    free(ws->activations);
    free(ws->deltas);

    ws->activations = NULL;
    ws->deltas     = NULL;
    ws->layers     = 0;
}


static void _forward(
    _Workspace *ws, 
    const Network *net, 
    const double *sample
){ // _forward(): compute activations layer by layer given one input sample
    memcpy(ws->activations[0], sample, net->layers[0].inputs * sizeof(double));
    
    for(size_t i = 0; i < net->n_layers; ++i){
        const Layer *layer = &net->layers[i];

        double *input  = ws->activations[i];
        double *output = ws->activations[i+1];

        for(size_t j=0; j < layer->neurons; ++j){
            double sum = layer->biases[j];
            // weights is a flattened [neurons x inputs] matrix, so neuron j's
            // weight row starts at offset j * layer->inputs.
            for(size_t k=0; k < layer->inputs; ++k){
                sum += layer->weights[j * layer->inputs + k] * input[k];
            }
            if(i == net->n_layers - 1)
                output[j] = sum;        // Linear Output
            else
                output[j] = _ReLU(sum); // Hidden Layers
        }
    }
}

static void _backprop(
    _Workspace *ws, 
    const Network *net, 
    const double *target
){
    const size_t last = ws->layers - 1;

    /* OUTPUT */
    const Layer *output = &net->layers[net->n_layers - 1];
    for(size_t i=0; i<output->neurons; ++i){
        const double pred = ws->activations[last][i];
        ws->deltas[last][i] = (pred - target[i]);
    }

    /* HIDDEN */

    // Walk backwards from the last hidden layer to the first (i == 0 is the
    // input layer and has no delta to compute, so the loop stops at i > 0).
    for(size_t i = last - 1; i > 0; --i){

        /* Layer connecting activation[i] -> activation[i+1] */
        const Layer *next_layer = &net->layers[i];

        const size_t current_neurons = next_layer->inputs;

        for(size_t j = 0; j < current_neurons; ++j){
            double sum = 0.0;

            // Error is propagated backwards through next_layer's weights,
            // accessed "transposed" (column j instead of row k) since we're
            // going output->input instead of input->output.
            for(size_t k = 0; k < next_layer->neurons; ++k)
                sum += ws->deltas[i+1][k] * next_layer->weights[k * next_layer->inputs + j];

            ws->deltas[i][j] = sum * _ReLU_derivative(ws->activations[i][j]); 
        }
    }
}


static TrainOptions MLP_DefaultTrainOptions(void){
    return (TrainOptions){
        .max_epochs = 1000,
        .learning_rate = 1e-3,
        .stop_loss = 1e-8,
        .verbose = false,
    };
}

static Dataset MLP_Create_Dataset(
    const double *samples, 
    double *output, 
    const size_t n_samples, 
    const size_t n_features,
    const size_t n_outputs
){
    if(!samples || n_samples == 0 || n_features == 0)
        return (Dataset){0};

    return (Dataset){
        samples, 
        output, 
        n_samples, 
        n_features,
        n_outputs
    };
}

static Network MLP_Create_Network(const size_t *topology, const size_t topology_size){
    Network net = {0};

    if(!topology || topology_size < 2)
        return net;

    for (size_t i = 0; i < topology_size; ++i)
        if (topology[i] == 0)
            return net;

    net.n_layers = topology_size - 1;

    net.layers = calloc(net.n_layers, sizeof *net.layers);
    if(!net.layers)
        return (Network){0};

    for(size_t i = 0; i < net.n_layers; ++i){
        Layer *layer = &net.layers[i];

        layer->inputs  = topology[i];
        layer->neurons = topology[i + 1];

        layer->weights = malloc(
            layer->neurons * 
            layer->inputs * 
            sizeof *layer->weights
        );

        layer->biases = malloc(
            layer->neurons * 
            sizeof *layer->biases
        );

        if(!layer->weights || !layer->biases)
            goto fail;

        for(size_t j = 0; j < layer->neurons; j++){
            for(size_t k = 0; k < layer->inputs; k++)
                layer->weights[j * layer->inputs + k] = _random_weight();
            layer->biases[j] = 0.0;
        }
    }

    return net;

    fail:
        MLP_Destroy_Network(&net);
        return (Network){0};
}

static void MLP_View_Network(const Network *net){
    if(!net || !net->layers || net->n_layers == 0)
        return;

    printf("\n");
    for(size_t i=0; i<net->n_layers; ++i){
        const Layer *layer = &net->layers[i];
        printf("\nLayer %zu (%zu -> %zu):\n",
            i, layer->inputs, layer->neurons
        );
        for(size_t j=0; j<layer->neurons; ++j){
            printf("\n  Neuron %zu:\n", j);
            printf("    W: [");
            for(size_t k=0; k<layer->inputs; ++k)
                printf("%8.3f ", layer->weights[j * layer->inputs + k]);
            printf(" ]\n");
            printf("    B:  %8.3f\n", layer->biases[j]);
        }
    }
    printf("\n");
}

static void MLP_View_Dataset(const Dataset *d){
    if(!d || !d->samples)
        return;

    printf("\nDataset Summary\n");
    printf("------------------------------\n");
    printf("Samples  : %zu\n", d->n_samples);
    printf("Features : %zu\n", d->n_features);
    printf("Outputs  : %zu\n\n", d->n_outputs);

    /* Header */
    for(size_t i = 0; i < d->n_features; ++i)
        printf("X%-7zu", i);

    if(d->output)
        for(size_t i = 0; i < d->n_outputs; ++i)
            printf("Y%-7zu", i);

    printf("\n");

    /* Data */
    for(size_t sample = 0; sample < d->n_samples; ++sample){

        for(size_t feat = 0; feat < d->n_features; ++feat)
            printf("%-8.3f",
                d->samples[sample * d->n_features + feat]);

        if(d->output){
            for(size_t out = 0; out < d->n_outputs; ++out)
                printf("%-8.3f",
                    d->output[sample * d->n_outputs + out]);
        }

        printf("\n");
    }

    printf("\n");
}

static bool MLP_Train(
    Network *net, 
    const Dataset *d,
    const TrainOptions *options
){
    if(!net || !net->layers || net->n_layers == 0)
        return false;
    if(!d || !d->samples || !d->output)
        return false;
    if(!options)
        return false;

    if(net->layers[0].inputs != d->n_features)
        return false;

    if(net->layers[net->n_layers - 1].neurons != d->n_outputs)
        return false;

    _Workspace ws = _Workspace_Create(net);

    if (!ws.activations)
        return false;
    
    double loss = 0.0;
    size_t epch = 0;

    for(size_t epoch = 1; epoch <= options->max_epochs; ++epoch){
        loss = 0.0; epch = epoch;

        for(size_t sample = 0; sample < d->n_samples; ++sample){
            const double *sample_ptr = d->samples + sample * d->n_features;

            _forward(&ws, net, sample_ptr);

            const double *target = d->output + sample * d->n_outputs;
            double *pred = ws.activations[ws.layers - 1];
            
            for(size_t i=0; i<d->n_outputs; ++i){
                double error = pred[i] - target[i];
                loss += error * error;
            }

            _backprop(&ws, net, target);

            for(size_t i = 0; i < net->n_layers; ++i){
                Layer *layer = &net->layers[i];

                for(size_t j = 0; j < layer->neurons; ++j){
                    for(size_t k = 0; k < layer->inputs; ++k){
                        // dL/dw[j][k] = delta of neuron j * activation feeding into it (input k)
                        double gradient = ws.deltas[i+1][j] * ws.activations[i][k];
                        layer->weights[j * layer->inputs + k] -= options->learning_rate * gradient;
                    }
                    layer->biases[j] -= options->learning_rate * ws.deltas[i+1][j];
                }
            }
        }

        loss /= (double)d->n_samples;

        if(options->verbose)
            _print_summary(epch, options->max_epochs, loss, NULL);

        if(loss <= options->stop_loss){
            if(options->verbose)
                _print_summary(epch, options->max_epochs, loss, "Stop loss reached");
            epch = 0; // signal: already printed the final summary above
            break;
        }
    }

    if(options->verbose && epch)
        _print_summary(epch, options->max_epochs, loss, "Maximum epochs reached");

    _Workspace_Destroy(&ws);
    return true;
}

static bool MLP_Predict_Dataset(const Network *net, const Dataset *d, double *buf){
    if(!net || !net->layers || net->n_layers == 0)
        return false;
    if(!d || !d->samples)
        return false;
    if(!buf)
        return false;

    if(net->layers[0].inputs != d->n_features)
        return false;

    if(net->layers[net->n_layers - 1].neurons != d->n_outputs)
        return false;

    _Workspace ws = _Workspace_Create(net);

    if(!ws.activations)
        return false;

    for(size_t sample = 0; sample < d->n_samples; ++sample){
        const double *sample_ptr = d->samples + d->n_features * sample;
        _forward(&ws, net, sample_ptr);

        double *pred = ws.activations[ws.layers - 1];
        for(size_t i=0; i<d->n_outputs; ++i)
            buf[sample * d->n_outputs + i] = pred[i];
    }

    _Workspace_Destroy(&ws);
    return true;
}

static void MLP_Destroy_Network(Network *net){
    if(!net || !net->layers)
        return;

    for(size_t i=0; i<net->n_layers; ++i){
            free(net->layers[i].weights);
            free(net->layers[i].biases);
        }
    free(net->layers);

    net->layers = NULL;
    net->n_layers = 0;
}

#endif /* MLP_IMPLEMENTATION */
#endif /* MLP_H */