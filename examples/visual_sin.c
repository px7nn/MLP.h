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

#define MLP_IMPLEMENTATION
#include "../MLP.h"

#define SIN_DATA  "../datasets/sinewave_train.csv"
#define PRED_DATA "../datasets/sinewave_pred.csv"

#define MAX_SAMPLES 1000
#define N_FEATURES  1
#define N_OUTPUTS   1

#define PI 3.14159265358979323846

bool Write_Pred(Network *n){
    FILE *fp = fopen(PRED_DATA, "w");
    if(!fp){
        perror("fopen");
        return false;
    }
    fprintf(fp, "x,pred(x)\n");

    double input[1], output[1];
    // Predict over [0, 4*PI].
    // Note: The training data only covers [0, 2*PI] (interpolation).
    // The range [2*PI, 4*PI] represents extrapolation, where predictions will fail to be periodic.
    for(double x=0; x <= 4 * PI; x+=0.01){
        input[0] = x;
        if(!MLP_Predict(n, input, output)){
            printf(
                "Cannot Predict: %s\n", MLP_ErrorString(MLP_GetLastError())
            );
            fclose(fp);
            return false;
        }
        fprintf(fp, "%lf,%lf\n", input[0], output[0]);
    }

    fclose(fp);
    return true;
}
bool GNUPLOT(){
    FILE *gp = popen("gnuplot -persistent", "w");
    if (!gp) {
        perror("gnuplot");
        return false;
    }

    fprintf(gp, "set datafile separator ','\n");
    fprintf(gp, "set title 'Neural Network Learning sin(x)'\n");
    fprintf(gp, "set xlabel 'x'\n");
    fprintf(gp, "set ylabel 'y'\n");
    fprintf(gp, "set grid\n");
    fprintf(gp, "set key top left\n");

    fprintf(gp,
        "plot "
        "'"SIN_DATA"' using 1:2 every ::1 "
        "with lines lw 2 lc rgb 'green' title 'Training Data', "
        "'"PRED_DATA"' using 1:2 every ::1 "
        "with lines lw 2 lc rgb 'red' title 'Prediction'\n"
    );

    fflush(gp);   // Send commands immediately

    /* Keep the window open */
    fprintf(gp, "pause -1\n");

    _pclose(gp);
    return true;
}

int main(){
    Dataset d = MLP_LoadCSV(
        SIN_DATA, MAX_SAMPLES, N_FEATURES, N_OUTPUTS, true
    );

    Network n = MLP_Create_Network(&(NetworkConfig){
        .topology       = (size_t[]){N_FEATURES, 32, 32, N_OUTPUTS},
        .topology_size  = 4,
        .activations    = (Activation[]) {ACT_SIGMOID, ACT_RELU, ACT_LINEAR},
        .initializers   = MLP_AUTO_INITIALIZERS,
        .loss           = LOSS_MSE
    });

    if(!n.layers){
        printf(
            "Cannot create network: %s\n", MLP_ErrorString(MLP_GetLastError())
        );
        goto destroy;
    }
    
    bool success = MLP_Train(&n, &d, &(TrainOptions){
        .max_epochs     = 5000,
        .learning_rate  = 1e-3,
        .stop_loss      = 1e-4,
        .verbose        = true
    });

    if(!success){
        printf(
            "Cannot train network: %s\n", MLP_ErrorString(MLP_GetLastError())
        );
        goto destroy;
    }

    Write_Pred(&n) && GNUPLOT();

    goto destroy;

    destroy:
        MLP_Destroy_Dataset(&d);
        MLP_Destroy_Network(&n);
        return MLP_GetLastError(); // returns 0 if counters no error
}