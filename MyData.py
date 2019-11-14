import keras


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

    @staticmethod
    def db(source):
        switcher = {
            'boston_housing': keras.datasets.boston_housing,
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
            'boston_housing': 13,
            'cifar10': 10,
            'cifar100': 100,
            'mnist': 10,
            'fashion_mnist': 10,
            'imdb': 2,
            'reuters': 46
        }
        return switcher.get(source, 10)

    def load(self):
        # Setup train and test splits
        if self.source == 'boston_housing' or self.source == 'mnist' or self.source == 'cifar10':
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
                                                                                  num_words=None, skip_top=0,
                                                                                  maxlen=None, seed=113, start_char=1,
                                                                                  oov_char=2, index_from=3)
        elif self.source == 'reuters':
            (x_train, y_train), (x_test, y_test) = self.db(self.source).load_data(path="reuters.npz",
                                                                                  num_words=None, skip_top=0,
                                                                                  maxlen=None, test_split=0.2,
                                                                                  seed=113, start_char=1, oov_char=2,
                                                                                  index_from=3)
            self.word_index = self.db(self.source).get_word_index(path="reuters_word_index.json")
        else:
            (x_train, y_train), (x_test, y_test) = self.db(self.source).load_data()

        print("Training label shape: ", y_train.shape)  # (60000,) -- 60000 numbers (all 0-9)
        print("First 5 training labels: ", y_train[:5])  # [5, 0, 4, 1, 9]

        # Flatten the images
        train_count = x_train.shape[0]
        test_count = x_test.shape[0]
        image_size = x_train.shape[1]
        image_vector_size = image_size * image_size
        if len(x_train.shape) == 4:
            channels = x_train.shape[3]
        else:
            channels = 1

        self.x_train = x_train
        self.x_test = x_test
        self.y_train_label = y_train.copy()
        self.y_test_label = y_test.copy()

        self.x_train1d = x_train.reshape(train_count, image_vector_size)
        self.x_test1d = x_test.reshape(test_count, image_vector_size)

        self.x_train2d = x_train.reshape(-1, image_size, image_size, channels)
        self.x_test2d = x_test.reshape(-1, image_size, image_size, channels)

        # Convert to "one-hot" vectors using the to_categorical function
        num_classes = self.db_categories(self.source)
        self.y_train = keras.utils.to_categorical(y_train, num_classes)
        self.y_test = keras.utils.to_categorical(y_test, num_classes)

        print("First 5 training labels as one-hot encoded vectors:\n", y_train[:5])
