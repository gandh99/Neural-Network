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

    # # Optional: See the breakdown of the dataset
    # indices = np.argmax(y, axis=1)
    # unique, counts = np.unique(indices, return_counts=True)
    # ans = dict(zip(unique, counts))
    # print(ans)

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
    accuracy, confusionMatrix, labelDict = evaluate_architecture(targets, preds)

    # Optional: Print results
    print(confusionMatrix)
    for i in range(len(labelDict)):
        key = "label" + str(i + 1)
        print(key, labelDict[key])
    print("Accuracy: ", accuracy)

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
    # Generate and populate the confusion matrix
    confusionMatrix = populate_confusion_matrix(y_true, y_pred)

    # Stores data on recall, precision and f1 for each label
    labelDict = dict.fromkeys({"label1", "label2", "label3", "label4"}) 

    # Compute and store the metrics 
    index = 0
    totalErrors = 0
    numOfRows = y_true.shape[0]
    for i in range(len(labelDict)):
        truePositive, falsePositive, falseNegative = calculate_metrics(confusionMatrix, index)
        recall = calculate_recall(truePositive, falseNegative)
        precision = calculate_precision(truePositive, falsePositive)
        f1 = calculate_f1(recall, precision)
        totalErrors += falsePositive

        key = "label" + str(index + 1)
        labelDict[key] = {"recall": recall, "precision": precision, "f1": f1}
        index += 1

    accuracy = calculate_classification_rate(numOfRows, totalErrors)

    # Return metrics
    return accuracy, confusionMatrix, labelDict
    
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
# E.g. [0, 1, 0, 0] returns 1
def extract_indices(data):
    indices = np.argmax(data, axis=1)

    return indices

# Metric: true positive, false positive, false negative
def calculate_metrics(matrix, index):
    numOfRows = matrix.shape[0]
    truePositive = matrix[index, index]
    falsePositive = 0
    falseNegative = 0

    # Calculate false positive and false negative
    for currentIndex in range(numOfRows):
        if currentIndex == index:
            continue
        falsePositive += matrix[index, currentIndex]
        falseNegative += matrix[currentIndex, index]    

    return truePositive, falsePositive, falseNegative

# Metric: recall = true pos / (true pos + false neg)
def calculate_recall(truePositive, falseNegative):
    if truePositive + falseNegative == 0:
        return 0
    return truePositive / (truePositive + falseNegative)

# Metric: precision = true pos / (true pos + false pos)
def calculate_precision(truePositive, falsePositive):
    if truePositive + falsePositive == 0:
        return 0
    return truePositive / (truePositive + falsePositive)

# Metric: F1 = 2 * (prec * rec) / (prec + rec)
def calculate_f1(precision, recall): 
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Metric: classification rate = 1 - classification error
def calculate_classification_rate(numOfRows, totalErrors):
    if numOfRows == 0:
        return 0
    return (numOfRows - totalErrors) / numOfRows


if __name__ == "__main__":
    # Setup for the hyperparameters for main()
    neurons = []
    activationFunctions = [] 
    outputDimension = 4

    # Modify any of the following hyperparameters     
    numOfHiddenLayers = 3              # Does not count input/output layer
    numOfNeuronsPerHiddenLayer = 20      # Configures all hidden layers to have the same number of neurons
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