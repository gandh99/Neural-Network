import numpy as np

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM
from sklearn.metrics import mean_squared_error


def main(_neurons, _activationFunctions, _batchSize, _learningRate, _numberOfEpochs):
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    input_dim = 3       # CONSTANT: Stated in specification
    neurons = _neurons
    activations = _activationFunctions
    net = MultiLayerNetwork(input_dim, neurons, activations)

    np.random.shuffle(dataset)

    x = dataset[:, :input_dim]
    y = dataset[:, input_dim:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=_batchSize,
        nb_epoch=_numberOfEpochs,
        learning_rate=_learningRate,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))
    print("Mean squared error:", evaluate_architecture(targets, preds))
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
    numOfHiddenLayers = 3               # Does not count input/output layer
    numOfNeuronsPerHiddenLayer = 5
    defaultActivation = "relu"          # Does not apply for input/output layer
    batchSize = 1000
    learningRate = 0.01
    numberOfEpochs = 1000

    # Optional: Set number of neurons in hidden layers based on hyperparameters
    # This results in all hidden layers having the same number of neurons (except output layer)
    for i in range(numOfHiddenLayers):
        neurons.append(numOfNeuronsPerHiddenLayer)
    neurons.append(3)       # CONSTANT: For the output layer

    # Optional: Set activation functions in hidden layers based on hyperparameters
    # This results in all hidden layers having the same activation functions (except output layer)
    for i in range(numOfHiddenLayers):
        activationFunctions.append(defaultActivation)
    activationFunctions.append("sigmoid")       # For the output layer

    # Call the main function to train and evaluate the neural network
    main(neurons, activationFunctions, batchSize, learningRate, numberOfEpochs)
