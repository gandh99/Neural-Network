import numpy as np

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)

from illustrate import illustrate_results_ROI


def main(_neurons, _activationFunctionHidden, _activationFunctionOutput, _lossFunction, _batchSize, _learningRate, _numberOfEpochs, _writeToCSV = False):
    dataset = np.loadtxt("ROI_dataset.dat")
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
    preds = net(x_val_pre)
    targets = y_val
    evaluate_architecture(targets, preds)
    # accuracy = evaluate_architecture(targets, preds)
    # print("Validation accuracy: {}".format(accuracy))

    # Optional: Write results to a csv file
    if _writeToCSV:
        with open('ROI_results.csv','a') as file:
            # No. of hidden layers, no. of neurons per hidden layer, activation, batch size, learning rate, number of epochs,
            # Accuracy
            csvList = [len(neurons) - 1, neurons[0], activations[0], _activationFunctionOutput, _batchSize, 
                _learningRate, _numberOfEpochs, accuracy]
            csvRow = str(csvList).strip("[]")
            csvRow += "\n"
            file.write(csvRow)
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_ROI(network, prep)

# First create the confusion matrix (predicted x expected)
# Then, evaluate the architecture using accuracy, precision, recall and F1 score
def evaluate_architecture(y_true, y_pred):
    confusionMatrix = populate_confusion_matrix(y_true, y_pred)
    print(confusionMatrix)

    # Test: Prints the number of occurrences of each index in y_true
    indices = np.argmax(y_true, axis=1)
    unique, counts = np.unique(indices, return_counts=True)
    ans = dict(zip(unique, counts))
    print(ans)
    
# Populates the confusion matrix based on y_true and y_pred
def populate_confusion_matrix(y_true, y_pred):
    # Create an empty confusion matrix filled with 0s
    numOfRows = y_pred.shape[1]
    numOfColumns = y_true.shape[1]
    confusionMatrix = create_confusion_matrix(numOfRows, numOfColumns)
    
    # Start populating the confusion matrix
    row = extract_indices(y_pred)
    col = extract_indices(y_true)
    for i in range(y_true.shape[0]):
        confusionMatrix[row[i], col[i]] += 1

    return confusionMatrix

# Create confusion matrix (predicted x expected) with all 0s
def create_confusion_matrix(numOfRows, numOfColumns):
    matrix = np.zeros(shape = (numOfRows, numOfColumns))

    return matrix

# Given a matrix of values, returns as an array the indices of the maximum value for each row
def extract_indices(data):
    indices = np.argmax(data, axis=1)

    return indices


if __name__ == "__main__":
    # Setup for the hyperparameters for main()
    neurons = []
    activationFunctions = [] 
    outputDimension = 4

    # Modify any of the following hyperparameters     
    numOfHiddenLayers = 3              # Does not count input/output layer
    numOfNeuronsPerHiddenLayer = 5      # Configures all hidden layers to have the same number of neurons
    activationHidden = "relu"          # Does not apply for input/output layer
    activationOutput = "sigmoid"
    lossFunction = "mse"
    batchSize = 64
    learningRate = 1e-3
    numberOfEpochs = 1000

    # Optional: Write results to csv
    writeToCSV = False

    # Optional: Set number of neurons in hidden layers based on hyperparameters
    # This results in all hidden layers having the same number of neurons (except output layer)
    for i in range(numOfHiddenLayers):
        neurons.append(numOfNeuronsPerHiddenLayer)
    neurons.append(outputDimension)       # CONSTANT: For the output layer

    # Optional: Set activation functions in hidden layers based on hyperparameters
    # This results in all hidden layers having the same activation functions (except output layer)
    for i in range(numOfHiddenLayers):
        activationFunctions.append(activationHidden)
    activationFunctions.append(activationOutput)       # For the output layer

    # Call the main function to train and evaluate the neural network
    main(neurons, activationFunctions, activationOutput, lossFunction, batchSize, learningRate, numberOfEpochs, writeToCSV)