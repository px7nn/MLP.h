/*=============================================================================
    MLP.h — A tiny single-header Multi-Layer Perceptron library in C

    A minimal, dependency-free feedforward neural network implementation
    supporting arbitrary topologies.

    Usage: define MLP_IMPLEMENTATION in exactly one translation unit before
    including this header to pull in the implementation.

    License: MIT (see LICENSE)
=============================================================================*/

#ifndef MLP_H
#define MLP_H

#define MLP_VERSION_MAJOR 0
#define MLP_VERSION_MINOR 5
#define MLP_VERSION_PATCH 0
#define MLP_VERSION_STRING "0.5.0"

#define MLP_MAGIC   0x4D4C5032u /* "MLP2" */
#define MLP_VERSION 2u


#ifndef MLP_CSV_LINE_BUFFER
#define MLP_CSV_LINE_BUFFER 1024    // Size of the temporary buffer used to read one CSV line
#endif

#define LEN(ARR) sizeof(ARR) / sizeof(ARR[0])


#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>


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


typedef enum {
    INIT_RANDOM,
    INIT_XAVIER,
    INIT_HE,

    INIT_COUNT
} Initializer;


typedef enum {
    ACT_LINEAR,
    ACT_RELU,
    ACT_LEAKY_RELU,
    ACT_SIGMOID,

    ACT_COUNT
} Activation;

typedef enum {
    LOSS_MSE,
    LOSS_BINARY_CROSS_ENTROPY,

    LOSS_COUNT
} Loss;


/*=============================================================================
    Public Data Structures
=============================================================================*/

typedef struct {
    const size_t *topology;
    const size_t topology_size;

    const Activation *activations;
    Loss loss;
    
    const Initializer initializer;
} NetworkConfig;

typedef struct {
    size_t max_epochs;
    double learning_rate;
    double stop_loss;
    bool verbose;
} TrainOptions;


typedef struct {
    double *samples; // Flattened: samples x features
    double *output;  // Flattened: samples x n_outputs (may be NULL for prediction-only datasets)

    size_t n_samples; 
    size_t n_features;
    size_t n_outputs;
} Dataset;

typedef struct {
    double *weights; // Flattened: neurons * inputs
    double *biases;  // neurons

    size_t neurons;  // number of neurons in this layer (its output width)
    size_t inputs;   // number of inputs into this layer (previous layer's neuron count)

    Activation activation;
} Layer;

typedef struct {
    Layer *layers;
    size_t n_layers;

    Loss loss;
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
    double *samples, 
    double *output, 
    size_t n_samples, 
    size_t n_features,
    size_t n_outputs
);

static TrainOptions MLP_DefaultTrainOptions(void);

static Network MLP_Create_Network(const NetworkConfig *cfg);
static void    MLP_View_Network(const Network *net);
static void    MLP_View_Dataset(const Dataset *d);
static bool    MLP_Train(Network *net, const Dataset *d, const TrainOptions *options);
static bool    MLP_Predict_Dataset(const Network *net, const Dataset *d, double *buf);
static void    MLP_Destroy_Network(Network *net);

static bool MLP_Save_Network(const Network *net, const char *filename);
static bool MLP_Load_Network(Network *net, const char *filename);

static const char *MLP_ErrorString(MLP_Error err);
static MLP_Error   MLP_GetLastError(void);

static Dataset MLP_LoadCSV(
    const char *filename,
    size_t max_samples,
    size_t n_features,
    size_t n_outputs,
    bool has_header
);
static void MLP_Destroy_Dataset(Dataset *d);

/*=============================================================================
    Private API
=============================================================================*/

static inline void _mlp_set_error(MLP_Error err);
static void _print_summary(size_t epoch, size_t max_epochs, double loss, const char *reason);

static inline double _sqrt(double x);
static inline double _pow(double base, unsigned int exp);
static inline double _exponential(double x); // to avoid using math.h
static inline double _initialize_weight(const size_t inputs, const Initializer initializer);

/* Activation funcs */
static inline double _ReLU(double z);
static inline double _Leaky_ReLU(double z);
static inline double _Sigmoid(double z);
static inline double _activation(double z, Activation act);


/* Derivative funcs */
static inline double _ReLU_derivative(double activation);
static inline double _Leaky_ReLU_derivative(double activation);
static inline double _Sigmoid_derivative(double activation);
static inline double _activation_derivative(double activation, Activation act);


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

static MLP_Error _mlp_last_error = MLP_OK;
 
static inline void _mlp_set_error(MLP_Error err){
    _mlp_last_error = err;
}

static void _exit_on_failure(){
    #ifdef MLP_EXIT_ON_ERROR
        fprintf(stderr, "MLP Error: %s\n", MLP_ErrorString(_mlp_last_error));
        exit(EXIT_FAILURE);
    #endif
}
 
static MLP_Error MLP_GetLastError(void){
    return _mlp_last_error;
}

static const char *MLP_ErrorString(MLP_Error err){
    switch(err){
        case MLP_OK:                      return "No error";
        case MLP_ERR_NULL_POINTER:        return "A required argument was NULL";
        case MLP_ERR_INVALID_ARGUMENT:    return "An argument was invalid";
        case MLP_ERR_ALLOC_FAILED:        return "Memory allocation failed";
        case MLP_ERR_SHAPE_MISMATCH:      return "Network and dataset dimensions are incompatible";
        case MLP_ERR_FILE_OPEN:           return "Could not open file";
        case MLP_ERR_FILE_READ:           return "Failed to read file (unexpected EOF or I/O error)";
        case MLP_ERR_FILE_WRITE:          return "Failed to write file (disk full or I/O error)";
        case MLP_ERR_FILE_FORMAT:         return "File is not a valid MLP model file";

        case MLP_ERR_CSV_EMPTY:           return "CSV file contains no data rows";
        case MLP_ERR_CSV_INVALID_NUMBER:  return "CSV contains an invalid numeric value";
        case MLP_ERR_CSV_COLUMN_COUNT:    return "CSV row has an incorrect number of columns";
        case MLP_ERR_CSV_LINE_TOO_LONG:   return "CSV line exceeds the maximum supported length";
        case MLP_ERR_CSV_HEADER:          return "CSV header is missing or invalid";

        default:                          return "Unknown error";
    }
}


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

static inline double _sqrt(double x){
    if(x <= 0.0f) return 0.0f;
    double y = x;
    for(size_t i=0; i<5; ++i)
        y = 0.5f * (y + x/y);
    return y;
}
static inline double _pow(double base, unsigned int exp){
    double res = 1.0;
    while(exp){
        if(exp & 1)
            res *= base;
        base *= base;
        exp >>= 1;
    }
    return res;
}

static inline double _exponential(double x){
    if(x > 709.0)
        return 1.7976931348623157e308;   /* Approx. DBL_MAX */
    if(x < -709.0)
        return 0.0;
    
    if (x < 0.0)
        return 1.0 / _exponential(-x);
    
    const double e = 2.71828182845904523536;

    unsigned int whole = (unsigned int)x;
    double frac = x - (double)whole;

    double res = _pow(e, whole);
    
    double term, sum;
    term = sum = 1.0;

    for(size_t i=1; i<=10; ++i){
        term *= frac / (double)i;
        sum += term;
    }
    return res * sum;
}
static inline double _initialize_weight(const size_t input_count, const Initializer initializer){
    double w= (((double)rand() / RAND_MAX) * 2.0 - 1.0);
    switch(initializer){
        case INIT_RANDOM: return w;
        case INIT_XAVIER: return _sqrt(1 / (double)input_count) * w;
        case INIT_HE:     return _sqrt(2 / (double)input_count) * w;

        default:          return w;
    }
    return w;
}

static inline double _ReLU(double z){
    return (z > 0.0)? z : 0;
}
static inline double _ReLU_derivative(double activation){
    return (activation > 0.0)? 1.0 : 0;
}

static inline double _Leaky_ReLU(double z){
    return (z > 0.0)? z : 0.01 * z;
}
static inline double _Leaky_ReLU_derivative(double activation){
    return (activation > 0.0)? 1.0 : 0.01;
}

static inline double _Sigmoid(double z){
    return 1.0 / (1.0 + _exponential(-z));
}
static inline double _Sigmoid_derivative(double activation){
    return activation * (1.0 - activation);
}


static inline double _activation(double z, Activation act){
    switch(act){
        case ACT_LINEAR:        return z;
        case ACT_RELU:          return _ReLU(z);
        case ACT_LEAKY_RELU:    return _Leaky_ReLU(z);
        case ACT_SIGMOID:       return _Sigmoid(z);
        default:                return z;
    }
    return z;
}

static inline double _activation_derivative(double activation, Activation act){
    switch(act){
        case ACT_LINEAR:        return 1.0;
        case ACT_RELU:          return _ReLU_derivative(activation);
        case ACT_LEAKY_RELU:    return _Leaky_ReLU_derivative(activation);
        case ACT_SIGMOID:       return _Sigmoid_derivative(activation);
        default:                return 1.0;
    }
    return 1.0;
}


static _Workspace _Workspace_Create(const Network *net){
    _Workspace ws = {0};

    if (!net || !net->layers || net->n_layers == 0)
        return ws;

    // +1 because activations/deltas include slot 0 for the raw input,
    // so activations[i+1]/deltas[i+1] line up with net->layers[i]'s output.
    ws.layers = net->n_layers + 1;

    ws.activations = calloc(ws.layers, sizeof *ws.activations);
    ws.deltas      = calloc(ws.layers, sizeof *ws.deltas);

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

        _mlp_set_error(MLP_ERR_ALLOC_FAILED);
        _exit_on_failure();
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
            output[j] = _activation(sum, layer->activation);
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
        
        switch(net->loss){
            case LOSS_MSE: 
                ws->deltas[last][i] = (pred - target[i]) * _activation_derivative(pred, output->activation);
                break;
            case LOSS_BINARY_CROSS_ENTROPY:
                ws->deltas[last][i] = (pred - target[i]);
                break;
            default:
                ws->deltas[last][i] = (pred - target[i]) * _activation_derivative(pred, output->activation);
                break;
        }
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

            ws->deltas[i][j] = sum * _activation_derivative(ws->activations[i][j], net->layers[i-1].activation); //current layer's activation
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
    double *samples, 
    double *output, 
    size_t n_samples, 
    size_t n_features,
    size_t n_outputs
){
    if(!samples){
        _mlp_set_error(MLP_ERR_NULL_POINTER);
        _exit_on_failure();
        return (Dataset){0};
    }
    if(n_samples == 0 || n_features == 0){
        _mlp_set_error(MLP_ERR_INVALID_ARGUMENT);
        _exit_on_failure();
        return (Dataset){0};
    }

    return (Dataset){
        samples, 
        output, 
        n_samples, 
        n_features,
        n_outputs
    };
}

static Network MLP_Create_Network(const NetworkConfig *cfg){
    Network net = {0};

    if(!cfg || !cfg->topology || !cfg->activations){
        _mlp_set_error(MLP_ERR_NULL_POINTER);
        _exit_on_failure();
        return net;
    }

    const size_t *topology = cfg->topology;
    const size_t n_layers  = cfg->topology_size - 1;

    if(cfg->topology_size < 2){
        _mlp_set_error(MLP_ERR_INVALID_ARGUMENT);
        _exit_on_failure();
        return net;
    }

    for (size_t i = 0; i < cfg->topology_size; ++i)
        if (topology[i] == 0){
            _mlp_set_error(MLP_ERR_INVALID_ARGUMENT);
            _exit_on_failure();
            return net;
        }
    for(size_t i = 0; i < n_layers; ++i)
        if(cfg->activations[i] >= ACT_COUNT){
            _mlp_set_error(MLP_ERR_INVALID_ARGUMENT);
            _exit_on_failure();
            return net;
        }
    
    if(cfg->loss >= LOSS_COUNT || cfg->initializer >= INIT_COUNT){
        _mlp_set_error(MLP_ERR_INVALID_ARGUMENT);
        _exit_on_failure();
        return net;
    }

    net.n_layers = n_layers;
    net.loss     = cfg->loss;

    net.layers = calloc(net.n_layers, sizeof *net.layers);
    if(!net.layers){
        _mlp_set_error(MLP_ERR_ALLOC_FAILED);
        _exit_on_failure();
        return (Network){0};
    }

    for(size_t i = 0; i < net.n_layers; ++i){
        Layer *layer = &net.layers[i];

        layer->inputs  = topology[i];
        layer->neurons = topology[i + 1];

        layer->activation = cfg->activations[i];

        layer->weights = calloc(
            layer->neurons * layer->inputs, 
            sizeof *layer->weights
        );

        layer->biases = calloc(
            layer->neurons, 
            sizeof *layer->biases
        );

        if(!layer->weights || !layer->biases)
            goto fail;

        for(size_t j = 0; j < layer->neurons; j++){
            for(size_t k = 0; k < layer->inputs; k++)
                layer->weights[j * layer->inputs + k] = _initialize_weight(layer->inputs, cfg->initializer);
        }
    }

    return net;

    fail:
        MLP_Destroy_Network(&net);
        _mlp_set_error(MLP_ERR_ALLOC_FAILED);
        _exit_on_failure();
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
    if(!net || !net->layers || net->n_layers == 0){
        _mlp_set_error(MLP_ERR_NULL_POINTER);
        _exit_on_failure();
        return false;
    }
    if(!d || !d->samples || !d->output){
        _mlp_set_error(MLP_ERR_NULL_POINTER);
        _exit_on_failure();
        return false;
    }
    if(!options){
        _mlp_set_error(MLP_ERR_NULL_POINTER);
        _exit_on_failure();
        return false;
    }
    if(net->layers[0].inputs != d->n_features){
        _mlp_set_error(MLP_ERR_SHAPE_MISMATCH);
        _exit_on_failure();
        return false;
    }
    if(net->layers[net->n_layers - 1].neurons != d->n_outputs){
        _mlp_set_error(MLP_ERR_SHAPE_MISMATCH);
        _exit_on_failure();
        return false;
    }

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
    if(!net || !net->layers || net->n_layers == 0){
        _mlp_set_error(MLP_ERR_NULL_POINTER);
        _exit_on_failure();
        return false;
    }
    if(!d || !d->samples){
        _mlp_set_error(MLP_ERR_NULL_POINTER);
        _exit_on_failure();
        return false;
    }
    if(!buf){
        _mlp_set_error(MLP_ERR_NULL_POINTER);
        _exit_on_failure();
        return false;
    }
    if(net->layers[0].inputs != d->n_features){
        _mlp_set_error(MLP_ERR_SHAPE_MISMATCH);
        _exit_on_failure();
        return false;
    }
    if(net->layers[net->n_layers - 1].neurons != d->n_outputs){
        _mlp_set_error(MLP_ERR_SHAPE_MISMATCH);
        _exit_on_failure();
        return false;
    }

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

    *net = (Network){0};
}


static bool MLP_Save_Network(const Network *net, const char *filename){
    if(!net || !net->layers || !filename){
        _mlp_set_error(MLP_ERR_NULL_POINTER);
        _exit_on_failure();
        return false;
    }

    FILE *fp = fopen(filename, "wb");
    if(!fp){
        _mlp_set_error(MLP_ERR_FILE_OPEN);
        _exit_on_failure();
        return false;
    }

    uint32_t magic   = MLP_MAGIC;
    uint32_t version = MLP_VERSION;

    if(fwrite(&magic,   sizeof magic,   1, fp) != 1)
        goto fail;
    if(fwrite(&version, sizeof version, 1, fp) != 1)
        goto fail;

    if(fwrite(&net->n_layers, sizeof net->n_layers, 1, fp) != 1)
        goto fail;

    if(fwrite(&net->loss, sizeof net->loss, 1, fp) != 1)
        goto fail;

    for(size_t i=0; i<net->n_layers; ++i){
        const Layer *layer = &net->layers[i];

        if(fwrite(&layer->neurons, sizeof layer->neurons, 1, fp) != 1)
            goto fail;
        if(fwrite(&layer->inputs,  sizeof layer->inputs,  1, fp) != 1)
            goto fail;

        if(fwrite(&layer->activation,  sizeof layer->activation,  1, fp) != 1)
            goto fail;


        if(fwrite(
            layer->weights, 
            sizeof *layer->weights, 
            layer->inputs * layer->neurons,
            fp
        ) != layer->inputs * layer->neurons) goto fail;

        if(fwrite(
            layer->biases,
            sizeof *layer->biases,
            layer->neurons,
            fp
        ) != layer->neurons) goto fail;
    }
    fclose(fp);
    return true;

    fail:
        fclose(fp);
        remove(filename);
        _mlp_set_error(MLP_ERR_FILE_WRITE);
        _exit_on_failure();
        return false;
}

static bool MLP_Load_Network(Network *net, const char *filename){
    if(!net || !filename){
        _mlp_set_error(MLP_ERR_NULL_POINTER);
        return false;
    }
    
    FILE *fp = fopen(filename, "rb");
    if(!fp){
        _mlp_set_error(MLP_ERR_FILE_OPEN);
        return false;
    }
    
    uint32_t magic;
    uint32_t version;

    if(fread(&magic,   sizeof magic,   1, fp) != 1){
        _mlp_set_error(MLP_ERR_FILE_READ);
        goto fail;
    }
    if(fread(&version, sizeof version, 1, fp) != 1){
        _mlp_set_error(MLP_ERR_FILE_READ);
        goto fail;
    }

    if(magic != MLP_MAGIC || version != MLP_VERSION){
        _mlp_set_error(MLP_ERR_FILE_FORMAT);
        goto fail;
    }

    MLP_Destroy_Network(net);

    if(fread(&net->n_layers, sizeof net->n_layers, 1, fp) != 1){
        _mlp_set_error(MLP_ERR_FILE_READ);
        goto fail;
    }

    if(fread(&net->loss, sizeof net->loss, 1, fp) != 1){
        _mlp_set_error(MLP_ERR_FILE_READ);
        goto fail;
    }
    if(net->loss >= LOSS_COUNT){
        _mlp_set_error(MLP_ERR_FILE_FORMAT);
        goto fail;
    }

    net->layers = calloc(net->n_layers, sizeof *net->layers);
    if(!net->layers){
        _mlp_set_error(MLP_ERR_ALLOC_FAILED);
        goto fail;
    }

    for(size_t i=0; i<net->n_layers; ++i){
        Layer *layer = &net->layers[i];

        if(fread(&layer->neurons, sizeof layer->neurons, 1, fp) != 1){
            _mlp_set_error(MLP_ERR_FILE_READ);
            goto fail;
        }
        if(fread(&layer->inputs,  sizeof layer->inputs,  1, fp) != 1){
            _mlp_set_error(MLP_ERR_FILE_READ);
            goto fail;
        }

        if(fread(&layer->activation,  sizeof layer->activation,  1, fp) != 1){
            _mlp_set_error(MLP_ERR_FILE_READ);
            goto fail;
        }
        if(layer->activation >= ACT_COUNT){
            _mlp_set_error(MLP_ERR_FILE_FORMAT);
            goto fail;
        }

        layer->weights = calloc(layer->neurons * layer->inputs, sizeof *layer->weights);
        layer->biases  = calloc(layer->neurons, sizeof *layer->biases);

        if(!layer->weights || !layer->biases){
            _mlp_set_error(MLP_ERR_ALLOC_FAILED);
            goto fail;
        }
        

        if(fread(
            layer->weights,
            sizeof *layer->weights,
            layer->neurons * layer->inputs,
            fp
        ) != layer->neurons * layer->inputs) {
            _mlp_set_error(MLP_ERR_FILE_READ);
            goto fail;
        }
        if(fread(
            layer->biases,
            sizeof *layer->biases,
            layer->neurons,
            fp
        ) != layer->neurons) {
            _mlp_set_error(MLP_ERR_FILE_READ);
            goto fail;
        }
    }
    fclose(fp);
    return true;

    fail:
        fclose(fp);
        MLP_Destroy_Network(net);
        _exit_on_failure();
        return false;
}


static Dataset MLP_LoadCSV(
    const char *filename,
    size_t max_samples,
    size_t n_features,
    size_t n_outputs,
    bool has_header
){
    Dataset d = {0};

    if(!filename){
        _mlp_set_error(MLP_ERR_NULL_POINTER);
        _exit_on_failure();
        return d;
    }
    if(n_features == 0 || max_samples == 0){
        _mlp_set_error(MLP_ERR_INVALID_ARGUMENT);
        _exit_on_failure();
        return d;
    }
    if(max_samples > SIZE_MAX / n_features / sizeof *d.samples){
        _mlp_set_error(MLP_ERR_INVALID_ARGUMENT);
        _exit_on_failure();
        return d;
    }
    if(n_outputs && max_samples > SIZE_MAX / n_outputs / sizeof *d.output){
        _mlp_set_error(MLP_ERR_INVALID_ARGUMENT);
        _exit_on_failure();
        return d;
    }

    FILE *fp = fopen(filename, "r");
    if(!fp){
        _mlp_set_error(MLP_ERR_FILE_OPEN);
        _exit_on_failure();
        return d;
    }

    d.samples = malloc(max_samples * n_features * sizeof *d.samples);
    if(!d.samples){
        _mlp_set_error(MLP_ERR_ALLOC_FAILED);
        goto fail;
    }

    if(n_outputs){
        d.output = malloc(max_samples * n_outputs * sizeof *d.output);
        if(!d.output){
            _mlp_set_error(MLP_ERR_ALLOC_FAILED);
            goto fail;
        }
    }

    char line[MLP_CSV_LINE_BUFFER];

    if(has_header && !fgets(line, sizeof line, fp)){
        _mlp_set_error(MLP_ERR_CSV_HEADER);
        goto fail;
    }

    size_t sample = 0;
    const size_t expected = n_features + n_outputs;

    while(sample < max_samples && fgets(line, sizeof line, fp)){

        if(!strchr(line, '\n') && !feof(fp)){
            _mlp_set_error(MLP_ERR_CSV_LINE_TOO_LONG);
            goto fail;
        }

        size_t col = 0;
        char *token = strtok(line, ",\n\r");

        while(token){
            char *end;
            double val = strtod(token, &end);

            if(end == token || *end != '\0'){
                _mlp_set_error(MLP_ERR_CSV_INVALID_NUMBER);
                goto fail;
            }

            if(col >= expected){
                _mlp_set_error(MLP_ERR_CSV_COLUMN_COUNT);
                goto fail;
            }

            if(col < n_features)
                d.samples[sample * n_features + col] = val;
            else
                d.output[sample * n_outputs + (col - n_features)] = val;

            ++col;
            token = strtok(NULL, ",\n\r");
        }

        if(col != expected){
            _mlp_set_error(MLP_ERR_CSV_COLUMN_COUNT);
            goto fail;
        }

        ++sample;
    }

    if(sample == 0){
        _mlp_set_error(MLP_ERR_CSV_EMPTY);
        goto fail;
    }

    if(ferror(fp)){
        _mlp_set_error(MLP_ERR_FILE_READ);
        goto fail;
    }

    if(sample < max_samples){
        double *tmp = realloc(d.samples, sample * n_features * sizeof *d.samples);
        if(tmp)
            d.samples = tmp;
    }

    if(n_outputs && sample < max_samples){
        double *tmp = realloc(d.output, sample * n_outputs * sizeof *d.output);
        if(tmp)
            d.output = tmp;
    }

    d.n_samples = sample;
    d.n_features = n_features;
    d.n_outputs = n_outputs;

    fclose(fp);
    return d;

    fail:
        free(d.samples);
        free(d.output);

        if(fp)
            fclose(fp);
        _exit_on_failure();
        return (Dataset){0};
}

static void MLP_Destroy_Dataset(Dataset *d){
    if (!d)
        return;

    free(d->samples);
    free(d->output);
    *d = (Dataset){0};
}

#endif /* MLP_IMPLEMENTATION */
#endif /* MLP_H */