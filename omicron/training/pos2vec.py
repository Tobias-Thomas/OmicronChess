import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from omicron.util.db_preprocess import load_matrix
import numpy as np
import os
import random
import math


def _init_autoencoder_model(layers=[773, 600, 400, 200, 100]):
    """ Inits the auto_encoder model with the given layers. All layers are currently dense layers.
    The optimizer is always adam and the loss is the binary crossentropy.

    Args:
        layers (list, optional): The list of units per layer in the autoencoder. The first entry
            has to be 773 and can only be changed when changing the representation of the bitboard

    Returns:
        model (tensorflow.keras.models.Model): The compiled model, ready for training
    """
    pos = Input(shape=(layers[0],), name='position')

    current = pos
    # Encoding
    for i,l in enumerate(layers[1:]):
        new_hidden = Dense(l, activation='relu', name='encode%d'%i)
        current = new_hidden(current)
    # Decoding
    for i,l in enumerate(reversed(layers[:-1])):
        new_hidden = Dense(l, activation='relu', name='decode%d'%i)
        current = new_hidden(current)

    model = Model(inputs=pos, outputs=current)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def train_pos2vec_model(parsed_path_white, parsed_path_black, batch_size, epoch_size=2e6,
                        num_epochs=200, white_share=0.5, layers=[773, 600, 400, 200, 100]):
    """ Trains the pos2vec model, by selecting random inputs from the two given paths every batch.

    Args:
        parsed_path_white (str): The path to the directory containing positions white wins.
            Currently only supports file of the type created with omicron.util.db_preprocess
        parsed_path_black (str): The path to the directory containing positions black wins.
            Currently only supports file of the type created with omicron.util.db_preprocess
        batch_size (int): The batch size to train on
        epoch_size (int, optional): The number of samples in each epoch. This should be a multiple
            of the batch_size.
        num_epochs (int, optional): The number of epochs to train on
        white_share (float, optional): The fraction of training data that comes from positions
            that are positive for white. This number can any be between 0 and 1
        layers (list, optional): The list of units per layer in the autoencoder. The first entry
            has to be 773 and can only be changed when changing the representation of the bitboard

    Returns:
        tensorflow.keras.models.Model: The trainined model
        tensorflow.keras.callbacks.History: The history of the training
    """
    model = _init_autoencoder_model(layers)
    seq = Pos2VecSequence(parsed_path_white, parsed_path_black, white_share, epoch_size, batch_size)
    with tf.device('/GPU:0'):
        history = model.fit(seq, epochs=num_epochs, verbose=1, workers=4, use_multiprocessing=True)
    return model, history


def save_encoder(model, save_name, save_dir, enc_layers=4):
    """ Saves the encoder part of a trained autoencoder to reuse it later in the deepchess Model.
    This function only saves the weights of the file, since they are the only needed part for
    this application.

    Args:
        model (tensorflow.keras.models.Model): A trained autoencoder model
        save_name (str): The name under which the model should get saved
        save_dir (str): Path to the directory where the weights get saved
        enc_layers (int, optional): The number of layers that got used for encoding. The input
            layer does not count. Needed to extract the encoding part of the model.
    """
    encoded = model.get_layer('encode%d'%(enc_layers-1)).output
    enc_model = Model(inputs=model.input, outputs=encoded)
    enc_model.save_weights(weight_dir+save_name)


class Pos2VecSequence(Sequence):
    def __init__(self, parsed_white, parsed_black, white_share, epoch_size, batch_size):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.parsed_white = parsed_white
        self.parsed_black = parsed_black
        self.white_share = white_share

    def __len__(self):
        return math.ceil(self.epoch_size / self.batch_size)

    def __getitem__(self, idx):
        random_inp = np.random.randint(0, 1, (self.batch_size,773), dtype=np.bool)
        return (random_inp, random_inp)
        batch_examples = np.zeros((self.batch_size,773))
        white_samples = int(self.batch_size * self.white_share)
        games = os.listdir(self.parsed_white)
        random_white_games = random.sample(games, white_samples)
        for g in random_white_games:
            random_game = load_matrix(self.parsed_white+g)
            batch_examples[i] = random_game[random.randint(0,9)]
        games = os.listdir(self.parsed_black)
        random_black_games = random.sample(games, self.batch_size-white_samples)
        for g in random_black_games:
            random_game = load_matrix(self.parsed_black+g)
            batch_examples[i] = random_game[random.randint(0,9)]
        return (batch_examples, batch_examples)
