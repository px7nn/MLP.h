/*
    Example: MNIST Handwritten Digit Classification

    This example trains an MLP on the MNIST dataset to classify handwritten digits.
    
    Dataset setup:
    1. Download the MNIST dataset in CSV format:
       - Train (60,000 samples): https://pjreddie.com/media/files/mnist_train.csv
       - Test (10,000 samples):  https://pjreddie.com/media/files/mnist_test.csv
    2. Place these files in your "datasets" directory as:
       - datasets/mnist_train.csv
       - datasets/mnist_test.csv

    Results:
        [########################################] 100% Epoch    50/   50  Loss 1.317e-02

        Training completed.

        Epochs     : 50
        Final Loss : 1.31719803e-02
        Reason     : Maximum epochs reached

        Score: 9771/10000
        Accuracy: 97.71%
*/


#define MLP_IMPLEMENTATION
#define MLP_CSV_LINE_BUFFER 16384
#define MLP_USE_LIBM
#include "../MLP.h"

#define TR_PATH "../datasets/mnist_train.csv"
#define TS_PATH "../datasets/mnist_test.csv"

void preprocess_dataset(Dataset *d){
    /*  swaps the d.samples[j * d.n_features + 0] <-> d.output[i] 
        and normalize each pixels */
    double *new_array = calloc(d->n_samples * 10, sizeof *new_array);
    for(size_t i=0; i<d->n_samples; ++i){
        // first col of sample is the label
        int label = (int)d->samples[i * d->n_features];

        // shift all elements to the left (prev col)
        memmove(
            &d->samples[i * d->n_features], 
            &d->samples[i * d->n_features + 1], 
            (d->n_features-1) * sizeof *d->samples
        );

        d->samples[i * d->n_features + (d->n_features-1)] = d->output[i]/255.0;
        new_array[i * 10 + label] = 1.0;

        for(size_t j=0; j<d->n_features-1; j++)
            d->samples[i * d->n_features + j] = d->samples[i * d->n_features + j]/255.0;
    }
    free(d->output);

    d->output    = new_array;
    d->n_outputs = 10;
}

int main(){
    Dataset trd = MLP_LoadCSV(TR_PATH, 60000, 28*28, 1, true);
    Dataset tsd = MLP_LoadCSV(TS_PATH, 10000, 28*28, 1, true);

    if(!trd.samples || !tsd.output){
        MLP_Perror("LoadCSV");
        goto destroy;
    }

    /*  MLP_LoadCSV() copies the csv into a buffer the last column is 
        interpreted as output.
        Therefore it needs to be processed and normalized before using. */
    
    preprocess_dataset(&trd);
    preprocess_dataset(&tsd);

    Network net = MLP_Create_Network(&(NetworkConfig){
        .topology      = (size_t[]){28*28, 128, 64, 10},
        .topology_size = 4,
        .activations   = (Activation[]){ACT_LEAKY_RELU, ACT_LEAKY_RELU, ACT_SOFTMAX},
        .initializers  = MLP_AUTO_INITIALIZERS,
        .loss          = LOSS_CATEGORICAL_CROSS_ENTROPY
    });

    if(!MLP_Train(&net, &trd, &(TrainOptions){
        .max_epochs    = 50,
        .learning_rate = 1e-2,
        .stop_loss     = 1e-4,
        .verbose       = true,
        .loss_file     = "./loss_log.csv"
    })){
        MLP_Perror("Train");
        goto destroy;
    };

    double *pred = malloc(tsd.n_samples * 10 * sizeof *pred);
    MLP_Predict_Dataset(&net, &tsd, pred);

    size_t correct = 0;
    for(size_t i=0; i<tsd.n_samples; ++i){
        size_t predicted = 0; // argmax indx in pred
        size_t target    = 0; // argmax indx in actual

        for(size_t j=1; j<10; ++j){
            if(pred[i * 10 + predicted] < pred[i * 10 + j])
                predicted = j;
            if(tsd.output[i * 10 + target] < tsd.output[i * 10 + j])
                target = j;
        }

        if(predicted == target)
            ++correct;
    }

    printf("Score: %zu/%zu\n", correct, tsd.n_samples);
    printf("Accuracy: %.2f%", (float)correct/tsd.n_samples * 100.0);

    goto destroy; 

    destroy:
        MLP_Destroy_Dataset(&trd);
        MLP_Destroy_Dataset(&tsd);
        free(pred);
        return MLP_GetLastError(); // will return 0 if no error was encountered
    return 0;
}