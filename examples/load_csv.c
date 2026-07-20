#define MLP_EXIT_ON_ERROR
#define MLP_IMPLEMENTATION
#include "../MLP.h"

#define DATA_PATH   "../datasets/circle.csv"
#define MAX_SAMPLE  30
#define N_FEATURES  2
#define N_OUTPUTS   1

int main(){
    Dataset d = MLP_LoadCSV(
        DATA_PATH,
        MAX_SAMPLE,
        N_FEATURES,
        N_OUTPUTS,
        true
    );

    Network n = MLP_Create_Network(&(NetworkConfig){
        .topology       = (size_t[]){2, 8, 8, 1},
        .topology_size  = 4,
        .activations    = (Activation[]){ACT_LEAKY_RELU, ACT_LEAKY_RELU, ACT_SIGMOID},
        .initializers   = MLP_AUTO_INITIALIZERS,
        .loss           = LOSS_BINARY_CROSS_ENTROPY
    });


    MLP_Train(&n, &d, &(TrainOptions){
        .max_epochs     = 10000,
        .learning_rate  = 1e-2,
        .stop_loss      = 1e-8,
        .verbose        = true
    });

    Dataset test = d;
    test.outputs = malloc(d.n_samples * N_OUTPUTS * sizeof *test.outputs);

    MLP_Predict_Dataset(&n, &test, test.outputs);

    MLP_View_Dataset(&test);

    free(test.outputs);  // Be careful cause this might never execute cause exit_on_failure
    MLP_Destroy_Dataset(&d);
}