/*
    Example: Loading a Saved Model

    This example loads the trained XOR model from "xor.mlp"
    and performs inference on the XOR dataset.

    NOTE:
        Run xor_gate.c first to train the network and create
        the "xor.mlp" model file.
*/

#define MLP_IMPLEMENTATION
#include "../MLP.h"

#define N_SAMPLES   4
#define N_FEATURES  2
#define N_OUTPUTS   1

/* XOR input samples */
double inputs[N_SAMPLES * N_FEATURES] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};

int main(void)
{
    /*--------------------------------------------------------------
        Create a dataset containing only the input samples.

        Since this example performs inference only, the output
        pointer is initially NULL. It will later point to the
        prediction buffer.
    --------------------------------------------------------------*/
    Dataset d = MLP_Create_Dataset(
        inputs,
        NULL,
        N_SAMPLES,
        N_FEATURES,
        N_OUTPUTS
    );

    /*--------------------------------------------------------------
        Create an empty network.

        MLP_Load_Network() will allocate and populate the network using
        the data stored in "xor.mlp".
    --------------------------------------------------------------*/
    Network net = {0};

    /* Load the previously trained model. */
    if (!MLP_Load_Network(&net, "xor.mlp")) {
        printf("Failed to load model: %s\n",
               MLP_ErrorString(MLP_GetLastError()));
        return 1;
    }

    /* Display the loaded network. */
    MLP_View_Network(&net);

    /* Allocate a buffer to receive the predicted outputs. */
    double predictions[N_SAMPLES * N_OUTPUTS];

    /* Store predictions in the buffer above. */
    d.output = predictions;

    
    /* Run inference on every input sample. */
    if (!MLP_Predict_Dataset(&net, &d, d.output)) {
        printf("Prediction failed: %s\n",
               MLP_ErrorString(MLP_GetLastError()));
        MLP_Destroy_Network(&net);
        return 1;
    }

    /* Display predictions. */
    MLP_View_Dataset(&d);

    /* Release all allocated memory. */
    MLP_Destroy_Network(&net);

    return 0;
}
