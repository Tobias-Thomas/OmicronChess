"""The deepchess module provides the possibility to train the full deepchess model and save it.

Before training the model you have to create the input data. This is done epoch by epoch. After
creating input data, you can train the model and save the whole model for later usage.

Typical Usage example:
    # first create train data
    create_deepchess_input('path/to/white', 'path/to/black', epoch_size, 'path/to/save')
    # this only creates the input data for one epoch, so you might call this more often
    model, history = train_deepchess_model('path/to/input', batch_size, epoch_size)
"""
import os
import random
import math
import string
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import numpy as np
from omicron.util.db_preprocess import load_matrix
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def _init_deepchess_model(enc_layers, comp_layers):
    pos_a = Input(shape=(enc_layers[0],), name='position1')
    pos_b = Input(shape=(enc_layers[0],), name='position2')
    # build the encoder part that uses shared weights
    current_a = pos_a
    current_b = pos_b
    for i, layer in enumerate(enc_layers[1:]):
        new_hidden = Dense(layer, activation='relu', name='encode%d' % i)
        current_a = new_hidden(current_a)
        current_b = new_hidden(current_b)

    # build the comparison part of the network
    merged_enc_positions = concatenate([current_a, current_b], axis=-1)
    current = merged_enc_positions
    for i, layer in enumerate(comp_layers):
        new_hidden = Dense(layer, activation='relu', name='comp%d' % i)
        current = new_hidden(current)

    model = Model(inputs=[pos_a, pos_b], outputs=current)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def _load_enc_weights(model, path_to_weights, num_enc_layers=4):
    weights = np.load(path_to_weights)
    weights = [weights['arr_%d' % i] for i in range(len(weights.files))]
    for i in range(num_enc_layers):
        model.get_layer('encode%d' % i).set_weights(weights[2*i: (2*i)+2])
    return model


def train_deepchess_model(input_dir, batch_size, epochs=1000,
                          path_to_enc_weights='pretrained_models/pos2vec/current.npz',
                          enc_layers=None, comp_layers=None):
    """Train the deepchess model, choosing a random dataset every epoch.

    The dataset gets randomly chosen from the input dir every new epoch. The reason for loading
    precomputed datasets instead of loading them while training is the time problem. You should
    use a batch_size that suits your graphic card.

    Args:
        input_dir (string): The directory that contains the input files. Those files should be
            created with the create_deepchess_input method
        batch_size (int): The batch size for training
        epochs (int, optional): The number of epochs for training
        path_to_enc_weights (str, optional): The path to the saved weights of the encoder model
        enc_layers (list, optional): The number of units per layer of the encoding part of
            the model. This has to fit to the shape of the presaved weights.
        comp_layers (list, optional): The number of units per layer of the comparison part
            of the model.
    Returns:
        tensorflow.keras.models.Model: The trainined model
        tensorflow.keras.callbacks.History: The history of the training
    """
    if not enc_layers:
        enc_layers = [773, 600, 400, 200, 100]
    if not comp_layers:
        comp_layers = [400, 200, 100, 2]
    model = _init_deepchess_model(enc_layers, comp_layers)
    model = _load_enc_weights(model, path_to_enc_weights, len(enc_layers)-1)

    seq = _DeepchessSequence(input_dir, batch_size=batch_size, epoch_size=int(1e6))
    with tf.device('/GPU:0'):
        history = model.fit(seq, epochs=epochs, verbose=1,
                            callbacks=[_OnEpochEnd([seq.on_epoch_end])])
    return model, history


def create_deepchess_input(parsed_path_white, parsed_path_black, epoch_size, save_dir):
    """Create the training input for one epoch of training of the deepchess model.

    Randomly draws two positions (one positive for white and one positive for black) from
    the two given directories and returns saves them in a random order. The index of the
    position that is favorable for white is the label. The 3 arrays are going to be saved
    with numpy.save_compressed and can be loaded with numpy.load.
    Currently this function has somewhat of a memory leak, going linearly up over time,
    converging to ~2GB at the end of the function call.

    Args:
        parsed_path_white (str): Path to the directory with positions favorable for white
        parsed_path_black (str): Path to the directory with positions favorable for black
        epoch_size (int): The number of samples that should be drawn from both sides
        save_dir (str): The Directory to save the 3 arrays to.
    """
    def create_one_input(path_white_game, path_black_game):
        white_game = load_matrix(path_white_game)
        white_pos = white_game[random.randint(0, min(9, len(white_game)-1))]
        black_game = load_matrix(path_black_game)
        black_pos = black_game[random.randint(0, min(9, len(black_game)-1))]

        white_win_index = np.random.choice([0, 1])
        if white_win_index == 0:
            return white_pos, black_pos, white_win_index
        return black_pos, white_pos, white_win_index

    white_games = random.choices(os.listdir(parsed_path_white), k=epoch_size)
    black_games = random.choices(os.listdir(parsed_path_black), k=epoch_size)
    input_a = np.zeros((epoch_size, 773), dtype=np.bool)
    input_b = np.zeros((epoch_size, 773), dtype=np.bool)
    labels = np.zeros((epoch_size, 2), dtype=np.bool)

    for i, (white, black) in enumerate(zip(white_games, black_games)):
        inp1, inp2, label_index = create_one_input(parsed_path_white+white,
                                                   parsed_path_black+black)
        input_a[i] = inp1
        input_b[i] = inp2
        labels[i, label_index] = 1

    save_name = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase +
                                      string.digits) for _ in range(10))
    np.savez_compressed(save_dir+save_name, input_a, input_b, labels)


class _DeepchessSequence(Sequence):
    """The Sequence of training data for the deepchess model.

    The purpose of this sequence is to preload a new dataset at the end of every epoch.
    """

    def __init__(self, input_dir, batch_size, epoch_size):
        self.input_dir = input_dir
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.input_a, self.input_b, self.labels = self._read_input()

    def __len__(self):
        return math.ceil(self.epoch_size / self.batch_size)

    def __getitem__(self, idx):
        batch_input_a = self.input_a[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_input_b = self.input_b[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_labels = self.labels[idx*self.batch_size: (idx+1)*self.batch_size]
        return [batch_input_a, batch_input_b], batch_labels

    def on_epoch_end(self):
        """Load a new dataset at the end of every epoch."""
        self.input_a, self.input_b, self.labels = self._read_input()

    def _read_input(self):
        inputs = os.listdir(self.input_dir)
        inp = np.load(self.input_dir+random.choice(inputs))
        return inp['arr_0'], inp['arr_1'], inp['arr_2']


class _OnEpochEnd(tf.keras.callbacks.Callback):
    # This class is only needed because of a tf bug that does not call the sequence on_epoch_end
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_epoch_end(self, epoch, logs=None):  # pylint: disable=W0613
        """Call all the passed functions at the end of each epoch."""
        for callback in self.callbacks:
            callback()
