#define MLP_EXIT_ON_ERROR
#define MLP_IMPLEMENTATION
#include "../MLP.h"

#define DATA_PATH   "../datasets/circle.csv"
#define MAX_SAMPLE  26
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

    NetworkConfig cfg = {
        .topology      = (size_t[]){2, 8, 8, 1},
        .topology_size = 4,
        .activations = (Activation[]){ACT_LEAKY_RELU, ACT_LEAKY_RELU, ACT_SIGMOID},
        .loss = LOSS_BINARY_CROSS_ENTROPY
    };

    Network n = MLP_Create_Network(&cfg);

    TrainOptions opt = MLP_DefaultTrainOptions();
    opt.verbose = true;
    opt.max_epochs = 10000;
    opt.learning_rate = 1e-2;

    MLP_Train(&n, &d, &opt);

    Dataset test = d;
    test.output = malloc(d.n_samples * N_OUTPUTS * sizeof *test.output);

    MLP_Predict_Dataset(&n, &test, test.output);

    MLP_View_Dataset(&test);

    free(test.output);  // Be careful cause this might never execute cause exit_on_failure
    MLP_Destroy_Dataset(&d);
}