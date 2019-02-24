import numpy as np

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)

from illustrate import illustrate_results_ROI
import matplotlib.pyplot as plt

# Global variables to stores values that will be used for plotting
xValues = []
yValues = []

# Main function that calls other functions to train and evaluate the neural network
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

    # Split data by rows into a training set and a validation set. We then augment the training data into the desired proportions
    # x_train = x[:split_idx]
    # y_train = y[:split_idx]
    augmentedTrainingData = augment_data(dataset[:split_idx, :], input_dim, label1=0.25, label2=0.25, label3=0.25, label4=0.25)
    x_train = augmentedTrainingData[:, :input_dim]
    y_train = augmentedTrainingData[:, input_dim:]
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

    # Optional: Append x and y values for plotting
    global xValues, yValues
    xValues.append(neurons[0])
    yValues.append(accuracy)

    # Optional: Write results to a csv file
    if _writeToCSV:
        with open('ROI_results.csv','a') as file:
            # No. of hidden layers, no. of neurons per hidden layer, activation in hidden layer, activation in output layer, 
            # batch size, learning rate, number of epochs, accuracy
            csvList = [len(neurons) - 1, neurons[0], activations[0], _activationFunctionOutput, _batchSize, 
                _learningRate, _numberOfEpochs, accuracy]
            csvRow = str(csvList).strip("[]")
            csvRow += "\n"
            file.write(csvRow)
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_ROI(network, prep)

# Augments the data into the desired proportion. The size of the new dataset will be the same as the input dataset
def augment_data(dataset, inputDim, label1=0.25, label2=0.25, label3=0.25, label4=0.25):
    # Calculate the relative proportions based on the input arguments
    label1 /= (label1 + label2 + label3 + label4)
    label2 /= (label1 + label2 + label3 + label4)
    label3 /= (label1 + label2 + label3 + label4)
    label4 /= (label1 + label2 + label3 + label4)
    listOfLabelProportions = [label1, label2, label3, label4]

    # Get the counts of each label in the input dataset and store as a dictionary (key = index, value = count)
    indices = np.argmax(dataset[:, inputDim:], axis=1)
    unique, counts = np.unique(indices, return_counts=True)
    countsDict = dict(zip(unique, counts)) 

    # Segregate the dataset according to the label
    numOfRows = dataset.shape[0]
    numOfColumns = dataset.shape[1]
    labelData = np.empty([0, numOfColumns])
    listOfLabelData = [labelData, labelData, labelData, labelData]      # Index 0 = label1 data, index 1 = label2 data, etc.
    for i in range(numOfRows):
        labelIndex = np.argmax(dataset[i, inputDim:])       # Get the index with the maximum value out of indices 0 to 3
        listOfLabelData[labelIndex] = np.append(listOfLabelData[labelIndex], [dataset[i, :]], axis=0)

    # # Sanity check to see if data was segregated correctly
    # print("Count dict:", countsDict)
    # print("shape:", listOfLabelData[0].shape)
    # print("shape:", listOfLabelData[1].shape)
    # print("shape:", listOfLabelData[2].shape)
    # print("shape:", listOfLabelData[3].shape)
    # print(listOfLabelData[3][0:10,:])

    # Augment the dataset
    newDataset = np.empty([0, numOfColumns])
    for i in range(len(listOfLabelData)):
        numOfDataNeeded = int(listOfLabelProportions[i] * numOfRows)
        if numOfDataNeeded <= listOfLabelData[i].shape[0]:
            newDataset = np.append(newDataset, listOfLabelData[i][:numOfDataNeeded, :], axis=0)
        else:
            numOfDuplicationsNeeded = int(numOfDataNeeded / listOfLabelData[i].shape[0])
            numOfRemaindersNeeded = int(numOfDataNeeded % listOfLabelData[i].shape[0])
            for j in range(numOfDuplicationsNeeded):
                newDataset = np.append(newDataset, listOfLabelData[i][:, :], axis=0)
            newDataset = np.append(newDataset, listOfLabelData[i][:numOfRemaindersNeeded, :], axis=0)

    # # Sanity check to see if the new dataset has the proportion we wanted
    # print("size of new dataset: ", newDataset.shape)
    # indices = np.argmax(newDataset[:, inputDim:], axis=1)
    # unique, counts = np.unique(indices, return_counts=True)
    # countsDict2 = dict(zip(unique, counts)) 
    # print(countsDict2)

    return newDataset   

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
    
# Populates the confusion matrix (predicted x expected) based on y_true and y_pred
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

# Plot a line graph of y against x
def plot_data(x, y):
    plt.plot(x, y, marker="x")
 
    # Set axes scales
    plt.ylim(0.5, 1.0)

    # Set label and title names
    xLabel = "Number of hidden layers"
    yLabel = "Accuracy"
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(yLabel + " vs " + xLabel)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    for iteratedValue in range(3, 6, 1):
        # Setup for the hyperparameters for main()
        neurons = []
        activationFunctions = [] 
        outputDimension = 4

        # Modify any of the following hyperparameters     
        numOfHiddenLayers = iteratedValue              # Does not count input/output layer
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

    # Optional: Plot the results in a line graph
    plot_data(xValues, yValues)