import keras
import numpy as np
from keras_preprocessing.text import Tokenizer


class Data:
    def __init__(self, source):
        self.source = source
        self.x_train = None
        self.x_test = None
        self.x_train2d = None
        self.x_test2d = None
        self.class_names = None
        self.word_index = None
        self.y_train = None
        self.y_test = None
        self.is_text = False

    @staticmethod
    def db(source):
        switcher = {
            'cifar10': keras.datasets.cifar10,
            'cifar100': keras.datasets.cifar100,
            'mnist': keras.datasets.mnist,
            'fashion_mnist': keras.datasets.fashion_mnist,
            'imdb': keras.datasets.imdb,
            'reuters': keras.datasets.reuters
        }
        return switcher.get(source, "mnist")

    @staticmethod
    def db_categories(source):
        switcher = {
            'cifar10': 10,
            'cifar100': 100,
            'mnist': 10,
            'fashion_mnist': 10,
            'imdb': 1,
            'reuters': 46
        }
        return switcher.get(source, 10)


    def __vectorize(self, sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1
        return results


    def load(self):
        # Setup train and test splits
        if self.source in ['mnist', 'cifar10']:
            (x_train, y_train), (x_test, y_test) = self.db(self.source).load_data()
        elif self.source == 'fashion_mnist':
                (x_train, y_train), (x_test, y_test) = self.db(self.source).load_data()
                x_train = x_train / 255.0
                x_test = x_test / 255.0
                self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        elif self.source == 'cifar100':
            (x_train, y_train), (x_test, y_test) = self.db(self.source).load_data(label_mode='fine')
        elif self.source == 'imdb':
            (x_train, y_train), (x_test, y_test) = self.db(self.source).load_data(path="imdb.npz",
                                                                                  num_words=10000, #memory limitation
                                                                                  skip_top=0,
                                                                                  maxlen=None, seed=113, start_char=1,
                                                                                  oov_char=2, index_from=3)

            data = np.concatenate((x_train, x_test), axis=0)
            targets = np.concatenate((y_train, y_test), axis=0)
            data = self.__vectorize(data)
            targets = np.array(targets).astype("float32")
            self.x_test = data[:10000]
            self.y_test = targets[:10000]
            self.x_train = data[10000:]
            self.y_train = targets[10000:]
            self.is_text = True

        elif self.source == 'reuters':
            (x_train, y_train), (x_test, y_test) = self.db(self.source).load_data(path="reuters.npz",
                                                                                  num_words=None, skip_top=0,
                                                                                  maxlen=None, test_split=0.2,
                                                                                  seed=113, start_char=1, oov_char=2,
                                                                                  index_from=3)
            self.word_index = self.db(self.source).get_word_index(path="reuters_word_index.json")
            self.is_text = True
        else:
            (x_train, y_train), (x_test, y_test) = self.db(self.source).load_data()

        print("Training label shape: ", y_train.shape)  # (60000,) -- 60000 numbers (all 0-9)
        print("First 5 training labels: ", y_train[:5])  # [5, 0, 4, 1, 9]

        # Flatten the images
        train_count = x_train.shape[0]
        test_count = x_test.shape[0]
        image_size = 0
        image_vector_size = 0
        if len(x_train.shape) >= 2:
            image_size = x_train.shape[1]
            image_vector_size = image_size * image_size

        if len(x_train.shape) == 4:
            channels = x_train.shape[3]
        else:
            channels = 1

        if self.source in ['mnist', 'cifar10', 'cifar100']:
            self.x_train = x_train
            self.x_test = x_test
            self.y_train_label = y_train.copy()
            self.y_test_label = y_test.copy()

            self.x_train1d = x_train.reshape(train_count, image_vector_size)
            self.x_test1d = x_test.reshape(test_count, image_vector_size)

            self.x_train2d = x_train.reshape(-1, image_size, image_size, channels)
            self.x_test2d = x_test.reshape(-1, image_size, image_size, channels)

            # Convert to "one-hot" vectors using the to_categorical function
            self.num_classes = self.db_categories(self.source)
            self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
            self.y_test = keras.utils.to_categorical(y_test, self.num_classes)
        elif self.source == 'reuters':
            self.num_classes = np.max(y_train) + 1
            max_words = 10000
            # Vectorizing sequence data
            tokenizer = Tokenizer(num_words=max_words)
            self.x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
            self.x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
            print('x_train shape:', x_train.shape)
            print('x_test shape:', x_test.shape)
            # Convert class vector to binary class matrix
            self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
            self.y_test = keras.utils.to_categorical(y_test, self.num_classes)

        print("First 5 training labels as one-hot encoded vectors:\n", y_train[:5])
