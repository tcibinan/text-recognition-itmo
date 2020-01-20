import time
import numpy as np

from scripts.common.utils import tanh, softmax, tanh2deriv


class NeuralNetwork:

    def __init__(self, num_letters, num_pixels, alpha=2, iterations=100, batch_size=100):
        self.num_letters = num_letters
        self.pixels_per_image = num_pixels
        self.alpha = alpha
        self.iterations = iterations
        self.batch_size = batch_size
        self.hidden_size = 512

    def train(self, x_train, y_train, seed=43):
        np.random.seed(seed)
        print('Train on %s samples' % x_train.shape[0])
        weights_0 = 0.02 * np.random.random((self.pixels_per_image, self.hidden_size)) - 0.01
        weights_1 = 0.2 * np.random.random((self.hidden_size, self.num_letters)) - 0.1
        for j in range(self.iterations):
            iteration_start = time.time()
            correct_cnt = 0
            for i in range(int(len(x_train) / self.batch_size)):
                batch_start, batch_end = ((i * self.batch_size), ((i + 1) * self.batch_size))
                layer_0 = x_train[batch_start:batch_end]
                layer_1 = tanh(np.dot(layer_0, weights_0))
                dropout_mask = np.random.randint(2, size=layer_1.shape)
                layer_1 *= dropout_mask * 2
                layer_2 = softmax(np.dot(layer_1, weights_1))

                for k in range(self.batch_size):
                    correct_cnt += int(
                        np.argmax(layer_2[k:k + 1]) == np.argmax(y_train[batch_start + k:batch_start + k + 1]))
                layer_2_delta = (y_train[batch_start:batch_end] - layer_2) / (self.batch_size * layer_2.shape[0])
                layer_1_delta = layer_2_delta.dot(weights_1.T) * tanh2deriv(layer_1)
                layer_1_delta *= dropout_mask
                weights_1 += self.alpha * layer_1.T.dot(layer_2_delta)
                weights_0 += self.alpha * layer_0.T.dot(layer_1_delta)
            iteration_end = time.time()
            print('Epoch %s/%s' %(str(j), self.iterations))
            print(' - %ss - val_accuracy: %s' % (np.around(iteration_end - iteration_start, 1), str(correct_cnt / float(len(x_train)))))
        np.savez('weights/weights.npz', weights_0=weights_0, weights_1=weights_1)

    def recognize(self, x_test):
        weights = np.load('weights/weights.npz')
        weights_0 = weights['weights_0']
        weights_1 = weights['weights_1']
        pred_result = []
        for i in range(len(x_test)):
            layer_0 = x_test[i:i + 1]
            layer_1 = tanh(np.dot(layer_0, weights_0))
            layer_2 = np.dot(layer_1, weights_1)
            pred_result.append(np.argmax(layer_2))
        return pred_result
