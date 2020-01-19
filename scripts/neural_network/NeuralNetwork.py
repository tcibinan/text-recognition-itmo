import numpy as np
from keras.datasets import mnist


class NeuralNetwork:

    def __init__(self, num_letters, num_pixels):
        self.num_letters = num_letters
        self.pixels_per_image = num_pixels
        self.weights_folder = "weights"
        self.alpha = 2
        self.iterations = 300
        self.hidden_size = 100
        self.batch_size = 100

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh2deriv(output):
        return 1 - (output**2)

    @staticmethod
    def softmax(x):
        temp = np.exp(x)
        return temp / np.sum(temp, axis=1, keepdims=True)

    def train(self, x_train, y_train):
        np.random.seed(1)
        weights_0_1 = 0.02 * np.random.random((self.pixels_per_image, self.hidden_size)) - 0.01
        weights_1_2 = 0.2 * np.random.random((self.hidden_size, self.num_letters)) - 0.1
        for j in range(self.iterations):
            correct_cnt = 0
            for i in range(int(len(x_train) / self.batch_size)):
                batch_start, batch_end = ((i * self.batch_size), ((i + 1) * self.batch_size))
                layer_0 = x_train[batch_start:batch_end]
                layer_1 = self.tanh(np.dot(layer_0, weights_0_1))
                dropout_mask = np.random.randint(2, size=layer_1.shape)
                layer_1 *= dropout_mask * 2
                layer_2 = self.softmax(np.dot(layer_1, weights_1_2))

                for k in range(self.batch_size):
                    correct_cnt += int(
                        np.argmax(layer_2[k:k + 1]) == np.argmax(y_train[batch_start + k:batch_start + k + 1]))
                layer_2_delta = (y_train[batch_start:batch_end] - layer_2) / (self.batch_size * layer_2.shape[0])
                layer_1_delta = layer_2_delta.dot(weights_1_2.T) * self.tanh2deriv(layer_1)
                layer_1_delta *= dropout_mask
                weights_1_2 += self.alpha * layer_1.T.dot(layer_2_delta)
                weights_0_1 += self.alpha * layer_0.T.dot(layer_1_delta)

            print("Epoch:" + str(j))
            print("Train-Acc:" + str(correct_cnt / float(len(x_train))))
        np.save(f"{self.weights_folder}/weights1", weights_0_1)
        np.save(f"{self.weights_folder}/weights2", weights_1_2)

    def recognize(self, x_test):
        weights_0_1 = np.load(f"{self.weights_folder}/weights1.npy")
        weights_1_2 = np.load(f"{self.weights_folder}/weights2.npy")
        pred_result = []
        for i in range(len(x_test)):
            layer_0 = x_test[i:i + 1]
            layer_1 = self.tanh(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)
            pred_result.append(np.argmax(layer_2))
        return pred_result


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    images, labels = (x_train[0:1000].reshape(1000, 28 * 28) / 255, y_train[0:1000])

    one_hot_labels = np.zeros((len(labels), 10))
    for i, l in enumerate(labels):
        one_hot_labels[i][l] = 1
    labels = one_hot_labels

    neural_network = NeuralNetwork(10, 28*28)
    neural_network.train(images, labels)

    x_test = x_test[:5]
    y_test = y_test[:5]
    test_images = x_test.reshape(len(x_test), 28 * 28) / 255
    test_labels = np.zeros((len(y_test), 10))
    for i, l in enumerate(y_test):
        test_labels[i][l] = 1

    neural_network.recognize(test_images)

