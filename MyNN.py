from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, \
    Conv1D  # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
from tensorflow import keras
import numpy as np

class Neural_net:
    def __init__(self):
        self.model = Sequential()
        self.conv = False

    def summary(self):
        self.model.summary()

    def add_input(self, neurons, activation_type, input_width, input_height, conv2d):
        if conv2d:
            self.conv = True
            # 32 convolution filters used each of size 3x3
            self.model.add(Conv2D(64, kernel_size=(3, 3), activation=activation_type, input_shape=(input_height, input_width, 1)))
            # choose the best features via pooling
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            # flatten since too many dimensions, we only want a classification output
            self.model.add(Flatten())
            self.model.add(Dense(units=neurons, activation=activation_type))
        else:
            self.model.add(Dense(units=neurons, activation=activation_type, input_shape=(input_width * input_height,)))

    def add_hidden(self, neurons, activation_type):
        self.model.add(Dense(units=neurons, activation=activation_type))

    def fit(self, x_train, y_train, x_test, y_test, batch, epoch, learning_rate, early_stop, drop_out, draw):
        if drop_out != 0:
            self.model.add(Dropout(drop_out/100))
        #if early_stop != 0:
        #    keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stop, verbose=0, mode='auto')


        epoch_callback = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: draw(epoch, logs)
        )

        cb = [LossAndErrorPrintingCallback(), epoch_callback]
        if early_stop != 0:
            cb.insert(0, EarlyStoppingAtMinLoss())

        sgd = keras.optimizers.SGD(lr=learning_rate)

        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(x_train, y_train, batch_size=batch, epochs=epoch, verbose=False, validation_split=.1, callbacks=cb)
        self.loss, self.accuracy = self.model.evaluate(x_test, y_test, verbose=False)
        print(f'Test accuracy: {self.accuracy:.3}')

class LossAndErrorPrintingCallback(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs=None):
    print('Accuracy {} .'.format(float(logs['accuracy'])))

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
  """Stop training when the loss is at its min, i.e. the loss stops decreasing.
  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

  def __init__(self, patience=0):
    super(EarlyStoppingAtMinLoss, self).__init__()
    self.patience = patience

    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None

  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    self.best = np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get('loss')
    if np.less(current, self.best):
      self.best = current
      self.wait = 0
      # Record the best weights if current results is better (less).
      self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        print('Restoring model weights from the end of the best epoch.')
        self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
