import numpy as np
import csv

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM
from sklearn.metrics import mean_squared_error


def main(_neurons, _activationFunctionHidden, _activationFunctionOutput, _lossFunction, _batchSize, _learningRate, _numberOfEpochs, _writeToCSV = False):
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    # Setup hyperparameters and neural network
    input_dim = 3       # CONSTANT: Stated in specification
    neurons = _neurons
    activations = _activationFunctionHidden
    net = MultiLayerNetwork(input_dim, neurons, activations)

    np.random.shuffle(dataset)

    # Separate data columns into x (input features) and y (output)
    x = dataset[:, :input_dim]
    y = dataset[:, input_dim:]

    split_idx = int(0.8 * len(x))

    # Split data by rows into a training set and a validation set
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    # Apply preprocessing to the data
    prep_input = Preprocessor(x_train)
    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=_batchSize,
        nb_epoch=_numberOfEpochs,
        learning_rate=_learningRate,
        loss_fun=_lossFunction,
        shuffle_flag=True,
    )

    # Train the neural network
    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    # Evaluate the neural network
    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    mse = evaluate_architecture(targets, preds)
    print("Validation accuracy: {}".format(accuracy))
    print("Mean squared error:", mse)

    if _writeToCSV:
        # Optional: Write results to a csv file
        with open('FM_results.csv','a') as file:
            # No. of hidden layers, no. of neurons per hidden layer, activation, batch size, learning rate, number of epochs,
            # Accuracy, MSE
            csvList = [len(neurons) - 1, neurons[0], activations[0], _activationFunctionOutput, _batchSize, 
                _learningRate, _numberOfEpochs, accuracy, mse]
            csvRow = str(csvList).strip("[]")
            csvRow += "\n"
            file.write(csvRow)
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_FM(net, prep)

def evaluate_architecture(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


if __name__ == "__main__":
    # Set hyperparameters for main()
    neurons = []
    activationFunctions = [] 

    # Modify any of the following hyperparameters     
    numOfHiddenLayers = 3              # Does not count input/output layer
    numOfNeuronsPerHiddenLayer = 5      # Configures all hidden layers to have the same number of neurons
    activationHidden = "relu"          # Does not apply for input/output layer
    activationOutput = "identity"
    lossFunction = "mse"
    batchSize = 1000
    learningRate = 1e-7
    numberOfEpochs = 1000

    # Optional: Write results to csv
    writeToCSV = False

    # Optional: Set number of neurons in hidden layers based on hyperparameters
    # This results in all hidden layers having the same number of neurons (except output layer)
    for i in range(numOfHiddenLayers):
        neurons.append(numOfNeuronsPerHiddenLayer)
    neurons.append(3)       # CONSTANT: For the output layer

    # Optional: Set activation functions in hidden layers based on hyperparameters
    # This results in all hidden layers having the same activation functions (except output layer)
    for i in range(numOfHiddenLayers):
        activationFunctions.append(activationHidden)
    activationFunctions.append(activationOutput)       # For the output layer

    # Call the main function to train and evaluate the neural network
    main(neurons, activationFunctions, activationOutput, lossFunction, batchSize, learningRate, numberOfEpochs, writeToCSV)
