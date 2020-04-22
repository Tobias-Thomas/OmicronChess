import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, concatenate
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


def _init_deepchess_model(enc_layers=[773,600,400,200,100], comp_layers=[400,200,100,2]):
    pos_a = Input(shape=(enc_layers[0],), name='position1')
    pos_b = Input(shape=(enc_layers[0],), name='position2')
    # build the encoder part that uses shared weights
    current_a = pos_a
    current_b = pos_b
    for i,l in enumerate(enc_layers[1:]):
        new_hidden = Dense(l, activation='relu', name='encode%d'%i)
        current_a = new_hidden(current_a)
        current_b = new_hidden(current_b)

    # build the comparison part of the network
    merged_enc_positions = concatenate([current_a, current_b], axis=-1)
    current = merged_enc_positions
    for i,l in enumerate(comp_layers):
        new_hidden = Dense(l, activation='relu', name='comp%d'%i)
        current = new_hidden(current)

    model = Model(inputs=[pos_a, pos_b], outputs=current)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def _load_enc_weights(model, path_to_weights, num_enc_layers=4):
    weights = np.load(path_to_weights)
    weights = [weights['arr_%d'%i] for i in range(len(weights.files))]
    for i in range(num_enc_layers):
        model.get_layer('encode%d'%i).set_weights(weights[2*i: (2*i)+2])
    return model


def train_deepchess_model(parsed_path_white, parsed_path_black, batch_size, epochs=100,
                          path_to_enc_weights='pretrained_models/pos2vec/current.npz',
                          enc_layers=[773, 600, 400, 200, 100], comp_layers=[400, 200, 100, 2]):
    model = _init_deepchess_model(enc_layers, comp_layers, path_to_enc_weights)
    model = _load_enc_weights(model, path_to_enc_weights, len(enc_layers)-1)

    seq = DeepchessSequence(parsed_path_white, parsed_path_black,
                            batch_size=batch_size, epoch_size=batch_size*100)
    with tf.device('/GPU:0'):
        model.fit(seq, epochs=epochs, verbose=1)
    return model, history


class  DeepchessSequence(Sequence):
    def __init__(self, parsed_white, parsed_black, epoch_size, batch_size):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.parsed_white = parsed_white
        self.parsed_black = parsed_black

    def __len__(self):
        return math.ceil(self.epoch_size / self.batch_size)

    def __getitem__(self, idx):
        white_games = os.listdir(self.parsed_white)
        black_games = os.listdir(self.parsed_black)
        batch_inp0 = np.zeros((self.batch_size,773))
        batch_inp1 = np.zeros((self.batch_size,773))
        labels = np.zeros((self.batch_size,2))
        for i in range(self.batch_size):
            random_white_game = load_matrix(self.parsed_white+random.choice(white_games))
            random_white_pos = random_white_game[random.randint(0,9)]
            random_black_game = load_matrix(self.parsed_black+random.choice(black_games))
            random_black_pos = random_black_game[random.randint(0,9)]

            white_pos_index = random.choice([0,1])
            labels[i,white_pos_index] = 1
            labels[i,1-white_pos_index] = 0

            batch_inp0[i] = random_black_pos if white_pos_index else random_white_pos
            batch_inp1[i] = random_white_pos if white_pos_index else random_black_pos

        return [batch_inp0, batch_inp1], labels
