/*
    Example: Visualizing Sine Wave Function Approximation

    This example trains an MLP to approximate the sin(x) function.
    It demonstrates two key machine learning concepts:
    
    1. Interpolation (x in [0, 2*PI]):
       Predicting output values within the range of the training data.
       The network should successfully learn and fit the sine wave curve here.

    2. Extrapolation (x in [2*PI, 4*PI]):
       Predicting output values outside the range of the training data.
       Since Multi-Layer Perceptrons lack periodic priors, they cannot
       extrapolate the periodic sine function. The predictions will deviate
       significantly (typically flatlining or continuing linearly) outside
       the training range [0, 2*PI].

    Requirements: gnuplot installed and in system PATH (optional, for visualization).
*/


#define MLP_EXIT_ON_ERROR
#define MLP_IMPLEMENTATION
#define _USE_MATH_DEFINES

#include <math.h>
#include "../MLP.h"

#define FN      sin      // Training function
#define FN_STR "sin"

#define TR    2 * M_PI   // Training range [0,2π]
#define TS    4 * M_PI   // Testing  range [0,4π]  

#define SAMPLES(range) ((size_t)(range/M_PI) * 100) 


Dataset init_data(double range, bool save_output);
bool GNUPLOT(double *in, double *pred);


int main(){
    Dataset TR_D = init_data(TR, true);
    Dataset TS_D = init_data(TS, false);

    Network n = MLP_Create_Network(&(NetworkConfig){
        .topology       = (size_t[]){1, 8, 1}, // 1->8->1
        .topology_size  = 3,
        .activations    = (Activation[]){ACT_TANH, ACT_LINEAR},
        .initializers   = MLP_AUTO_INITIALIZERS,
        .loss           = LOSS_MSE
    });
    MLP_Train(&n, &TR_D, &(TrainOptions){
        .max_epochs     = 5000,
        .stop_loss      = 1e-12,
        .learning_rate  = 1e-2,
        .verbose        = true 
    });

    MLP_Predict_Dataset(&n, &TS_D, TS_D.outputs);

    GNUPLOT(TS_D.inputs, TS_D.outputs);

    MLP_Destroy_Dataset(&TR_D);
    MLP_Destroy_Network(&n);
    MLP_Destroy_Dataset(&TS_D);
    return 0;
}

Dataset init_data(double range, bool save_output){
    const size_t n_samples = SAMPLES(range);
    Dataset d = {
        .inputs     = calloc(n_samples, sizeof *d.inputs),
        .outputs    = calloc(n_samples, sizeof *d.outputs),
        .n_samples  = n_samples,
        .n_features = 1,
        .n_outputs  = 1
    };
    if(!d.inputs || !d.outputs){
        perror("malloc");
        return (Dataset){0};
    }

    const double step = range/(n_samples - 1);
    for(size_t i=0; i<n_samples; ++i){
        double x = i * step;
        d.inputs[i] =  x;
        if(save_output)
            d.outputs[i] = FN(x);
    }
    return d;
}

bool GNUPLOT(double *in, double *pred){
    size_t n_samples = SAMPLES(TS);
    FILE *g = popen("gnuplot", "w");
    if(!g){
        perror("gnuplot");
        return false;
    }

    fprintf(g, "set terminal pngcairo size 800,500 background rgb '#121212' font 'sans,10'\n");
    fprintf(g, "set output '../docs/interpolation_vs_extrapolation.png'\n");

    /* purely for aesthetic purpose :) */
    fprintf(g, "set border lc rgb '#555555'\n");
    fprintf(g, "set grid lc rgb '#333333'\n");
    fprintf(g, "set xrange [0:%f]\n", TS);
    fprintf(g, "set yrange [-2.5:2.5]\n");
    fprintf(g, "set xtics tc rgb '#cccccc'\n");
    fprintf(g, "set ytics tc rgb '#cccccc'\n");
    fprintf(g, "set title 'Neural Network Interpolation vs Extrapolation' tc rgb '#ffffff' font 'sans,12,bold'\n");
    fprintf(g, "set ylabel '%s(x)' tc rgb '#cccccc'\n", FN_STR);
    fprintf(g, "set xlabel 'Input x' tc rgb '#cccccc'\n");
    fprintf(g, "set key top right tc rgb '#ffffff'\n");

    /* vertical line at x = TR */
    fprintf(g, 
        "set arrow from %f, graph 0 to %f, "
        "graph 1 nohead lc rgb '#ff5555' dt 2 lw 2\n",
        TR, TR
    );
    fprintf(g, "set label 'Training Range' at %f, graph 0.9 center tc rgb '#ff5555'\n", TR / 2.0);
    fprintf(g, "set label 'Extrapolation Range' at %f, graph 0.9 center tc rgb '#888888'\n", TR + (TS - TR) / 2.0);

    /* plot FN(x) then prediction */
    fprintf(g,
        "plot %s(x) title '%s(x) (Target)' with lines lw 2 lc rgb '#00aa00', " 
        "'-' using 1:2 with lines lw 2 lc rgb '#ffaa00' title 'Prediction'\n",
        FN_STR, FN_STR
    );

    for(size_t i=0; i<n_samples; ++i){
        fprintf(g, "%lf %lf\n", in[i], pred[i]);
    }

    fprintf(g, "e\n");

    fflush(g);
    pclose(g);
    return true;
}