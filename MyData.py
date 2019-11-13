import keras
from keras import utils
from keras.datasets import mnist



class Data:
    def __init__(self, source):
        self.source = source
        self.x_train = None
        self.x_test = None

    def load(self):
        # Setup train and test splits
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print("Training label shape: ", y_train.shape) # (60000,) -- 60000 numbers (all 0-9)
        print("First 5 training labels: ", y_train[:5]) # [5, 0, 4, 1, 9]

        # Flatten the images
        image_vector_size = 28 * 28
        self.x_train = x_train.reshape(x_train.shape[0], image_vector_size)
        self.x_test = x_test.reshape(x_test.shape[0], image_vector_size)

        self.x_train2d = self.x_train.reshape(-1, 28, 28, 1)
        self.x_test2d = self.x_test.reshape(-1, 28, 28, 1)

        # Convert to "one-hot" vectors using the to_categorical function
        num_classes = 10
        self.y_train = keras.utils.to_categorical(y_train, num_classes)
        self.y_test = keras.utils.to_categorical(y_test, num_classes)
        print("First 5 training lables as one-hot encoded vectors:\n", y_train[:5])