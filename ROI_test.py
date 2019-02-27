# File to test the predict_hidden function

from learn_ROI import *

if __name__ == '__main__':
    dataset = np.loadtxt("ROI_dataset.dat")
    np.random.shuffle(dataset)
    numOfRows = int(0.8*dataset.shape[0])
    output = predict_hidden(dataset[:numOfRows, :])
    print(output)