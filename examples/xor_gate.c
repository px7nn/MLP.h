/* 
    Example: XOR Gate Training 

    This example trains a Multi-Layer Perceptron (MLP) to learn the 
    XOR logic gate using backpropagation. 
    
    After training, the model is used to predict the XOR outputs and 
    is then saved to "xor.mlp". 
    
    NOTE: Run load_model.c after this example to load the saved model
    and perform inference without retraining. 
*/

#define MLP_IMPLEMENTATION
#include "../MLP.h"

#include <stdio.h>

#define N_SAMPLES   4
#define N_FEATURES  2
#define N_OUTPUTS   1

/* XOR training inputs */
static double inputs[] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};

/* Expected XOR outputs */
static double targets[] = {
    0,
    1,
    1,
    0
};

int main(void) {
    /* Create the training dataset. */
    Dataset train = MLP_Create_Dataset(
        inputs,
        targets,
        N_SAMPLES,
        N_FEATURES,
        N_OUTPUTS
    );

    /* Create a 2 → 5 → 1 network. */
    const size_t topology[] = {
        N_FEATURES,
        5,
        N_OUTPUTS
    };

    Network net = MLP_Create_Network(
        topology,
        sizeof(topology) / sizeof(topology[0])   // Number of entries in topology[]
    );

    /* Configure training. */
    TrainOptions opt = MLP_DefaultTrainOptions();
    opt.learning_rate = 1e-1;
    opt.verbose = true;

    /* Train the network. */
    if (!MLP_Train(&net, &train, &opt)) {
        printf("Training failed: %s\n",
            MLP_ErrorString(MLP_GetLastError()));
        return 1;
    }

    /* Predict every sample. */
    double predictions[N_SAMPLES * N_OUTPUTS];

    Dataset test = MLP_Create_Dataset(
        inputs,
        predictions,
        N_SAMPLES,
        N_FEATURES,
        N_OUTPUTS
    );

    if (!MLP_Predict_Dataset(&net, &test, predictions)) {
        printf("Prediction failed: %s\n",
            MLP_ErrorString(MLP_GetLastError()));
        MLP_Destroy_Network(&net);
        return 1;
    }

    /* Display predictions. */
    MLP_View_Dataset(&test);

    /* Save the trained model. */
    if (!MLP_Save(&net, "xor.mlp")) {
        printf("Save failed: %s\n",
            MLP_ErrorString(MLP_GetLastError()));
        MLP_Destroy_Network(&net);
        return 1;
    }

    printf("Model saved to xor.mlp\n");

    /* Release all allocated memory. */
    MLP_Destroy_Network(&net);

    return 0;
}