import numpy as np
from scipy.special import softmax

class ELMClassifier:
    """Probabilistic Output Extreme Learning Machine"""
    def __init__(self, hidden_layer_size=5):
        self.hidden_layer_size = hidden_layer_size

    # This function is used for training the ELM model
    def fit(self, x, y, c=1):
        # Thresholding the labels to avoid numerical instability
        y[y<0.5] = 0.0001
        y[y>0.5] = 0.9999

        # Get the number of features in the input and output
        x_features, y_features = x.shape[1], y.shape[1]

        # Initialize hidden neurons with random weights and biases
        self.hidden_neurons = [(np.random.randn(x_features), np.random.randn(1)) for i in range(self.hidden_layer_size)]

        # Compute the hidden layer output matrix H
        self.H = np.asarray([self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T

        # Compute the H'H matrix (H transpose multiplied by H)
        hth = np.dot(np.transpose(self.H), self.H)

        # Compute the inverse of (H'H + I/c), where I is the identity matrix
        inv_hth_plus_ic = np.linalg.pinv(hth + np.eye(hth.shape[0]) / c)

        # Compute the H' multiplied by the log-likelihood of the target labels
        ht_logs = np.dot(np.transpose(self.H), np.log(1 - y) - np.log(y))

        # Compute the output weights beta
        self.beta = -1 * np.dot(inv_hth_plus_ic, ht_logs)

    # This function is used for predicting class labels
    def predict(self, x):
        # Compute the hidden layer output matrix for test samples
        h = np.asarray([self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T

        # Compute the probability estimates using the output weights beta
        ret = 1.0 / (1 + np.exp(-1 * np.dot(h, self.beta)))

        # Normalize the probability estimates
        sums = np.sum(ret, axis=1)
        ret1 = ret / sums.reshape(-1, 1)
        ret2 = softmax(ret, axis=-1)
        retfinal = np.ones(ret.shape)
        retfinal[sums >= 1, :] = ret1[sums >= 1, :]
        retfinal[sums < 1, :] = ret2[sums < 1, :]

        # Return the class with the highest probability
        return np.argmax(retfinal, axis=-1)

    # This function is used for predicting class probabilities
    def predict_proba(self, x):
        # Compute the hidden layer output matrix for test samples
        h = np.asarray([self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T

        # Compute the probability estimates using the output weights beta
        ret = 1.0 / (1 + np.exp(-1 * np.dot(h, self.beta)))

        # Normalize the probability estimates
        sums = np.sum(ret, axis=1)
        ret1 = ret / sums.reshape(-1, 1)
        ret2 = softmax(ret, axis=-1)
        retfinal = np.ones(ret.shape)
        retfinal[sums >= 1, :] = ret1[sums >= 1, :]
        retfinal[sums < 1, :] = ret2[sums < 1, :]
        return retfinal


    def _activate(self, a, x, b):
        return 1.0 / (1 + np.exp(-1 * np.dot(a, x.T) + b))


