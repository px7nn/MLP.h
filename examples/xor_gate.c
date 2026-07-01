#define MLP_IMPLEMENTATION
#include "../MLP.h"

#define N_SAMPLES  4
#define N_FEATURES 2
#define N_OUTPUTS  1

/* XOR training inputs */
double inputs[N_SAMPLES * N_FEATURES] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};

/* Expected XOR outputs */
double out[N_SAMPLES * N_OUTPUTS] = {
    0,
    1,
    1,
    0
};

int main(void)
{
    /*--------------------------------------------------------------
        Create the training dataset.
        The samples and expected outputs are stored as flattened arrays.
    --------------------------------------------------------------*/
    Dataset training_dataset = MLP_Create_Dataset(
        inputs,
        out,
        N_SAMPLES,
        N_FEATURES,
        N_OUTPUTS
    );

    MLP_View_Dataset(&training_dataset);

    /*--------------------------------------------------------------
        Create a network with topology:

            2 input neurons
                  ↓
            5 hidden neurons
                  ↓
            1 output neuron
    --------------------------------------------------------------*/
    size_t topology[] = {N_FEATURES, 5, N_OUTPUTS};

    Network n = MLP_Create_Network(
        topology,
        sizeof(topology) / sizeof(topology[0])
    );

    /*--------------------------------------------------------------
        Start from the default training options and modify only the
        settings we care about.
    --------------------------------------------------------------*/
    TrainOptions opt = MLP_DefaultTrainOptions();
    opt.verbose       = true;
    opt.learning_rate = 1e-1;

    /*--------------------------------------------------------------
        Train the network using backpropagation.
    --------------------------------------------------------------*/
    if (!MLP_Train(&n, &training_dataset, &opt)) {
        printf("Cannot train network.\n");
        MLP_Destroy_Network(&n);
        return 1;
    }

    /*--------------------------------------------------------------
        Buffer that will receive the predicted outputs.
    --------------------------------------------------------------*/
    double buf[N_SAMPLES * N_OUTPUTS];

    /*--------------------------------------------------------------
        Create a prediction dataset.

        Only the input samples are required.
        The output pointer points to our prediction buffer.
    --------------------------------------------------------------*/
    Dataset testing_dataset = MLP_Create_Dataset(
        inputs,
        buf,
        N_SAMPLES,
        N_FEATURES,
        N_OUTPUTS
    );

    /*--------------------------------------------------------------
        Run inference on every sample.
    --------------------------------------------------------------*/
    if (!MLP_Predict_Dataset(&n, &testing_dataset, testing_dataset.output)) {
        printf("Prediction failed.\n");
        MLP_Destroy_Network(&n);
        return 1;
    }

    /*--------------------------------------------------------------
        Display the predicted outputs.
    --------------------------------------------------------------*/
    MLP_View_Dataset(&testing_dataset);

    /*--------------------------------------------------------------
        Free all memory allocated by the network.
    --------------------------------------------------------------*/
    MLP_Destroy_Network(&n);

    return 0;
}