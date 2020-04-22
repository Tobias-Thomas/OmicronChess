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
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

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


def train_pos2vec_model(parsed_path_white, parsed_path_black, batch_size, num_epochs=200,
                        epoch_size=int(2e6), white_share=0.5, layers=[773, 600, 400, 200, 100]):
    """ Trains the pos2vec model, by selecting random inputs from the two given paths every batch.

    Args:
        parsed_path_white (str): The path to the directory containing positions white wins.
            Currently only supports file of the type created with omicron.util.db_preprocess
        parsed_path_black (str): The path to the directory containing positions black wins.
            Currently only supports file of the type created with omicron.util.db_preprocess
        batch_size (int): The batch size to train on
        epoch_size (int, optional): The number of samples in each epoch
        num_epochs (int, optional): The number of epochs to train on
        white_share (float, optional): The fraction of training data that comes from positions
            that are positive for white. This number can any be between 0 and 1
        layers (list, optional): The list of units per layer in the autoencoder. The first entry
            has to be 773 and can only be changed when changing the representation of the bitboard

    Returns:
        tensorflow.keras.models.Model: The trainined model
        tensorflow.keras.callbacks.History: The history of the training
    """
    def load_training_samples():
        white_samples = int(epoch_size * white_share)
        white_games = os.listdir(parsed_path_white)
        black_games = os.listdir(parsed_path_black)
        examples = np.zeros((epoch_size,773))
        for i in range(white_samples):
            random_game = random.choice(white_games)
            random_game = load_matrix(parsed_path_white+random_game)
            examples[i] = random_game[random.randint(0,min(9, random_game.shape[0]-1))]
        for i in range(white_samples,epoch_size):
            random_game = random.choice(black_games)
            random_game = load_matrix(parsed_path_black+random_game)
            examples[i] = random_game[random.randint(0,min(9, random_game.shape[0]-1))]
        return (examples, examples)

    samples, labels = load_training_samples()
    model = _init_autoencoder_model(layers)
    with tf.device('/GPU:0'):
        history = model.fit(samples, labels, epochs=num_epochs, batch_size=batch_size, verbose=1)
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
    enc_weights = [x.numpy() for x in model.trainable_variables[:enc_layers*2]]
    np.savez(save_dir+save_name, *enc_weights)
