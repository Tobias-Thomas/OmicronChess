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
import string
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


def train_deepchess_model(input_dir, batch_size, epochs=1000,
                          path_to_enc_weights='pretrained_models/pos2vec/current.npz',
                          enc_layers=[773, 600, 400, 200, 100], comp_layers=[400, 200, 100, 2]):
    model = _init_deepchess_model(enc_layers, comp_layers)
    model = _load_enc_weights(model, path_to_enc_weights, len(enc_layers)-1)

    seq = DeepchessSequence(input_dir, batch_size=batch_size, epoch_size=int(1e6))
    with tf.device('/GPU:0'):
        history = model.fit(seq, epochs=epochs, verbose=1,
                            callbacks=[OnEpochEnd([seq.on_epoch_end])])
    return model, history


def create_deepchess_input(parsed_path_white, parsed_path_black, epoch_size, save_dir):
    def create_one_input(path_white_game, path_black_game):
        white_game = load_matrix(path_white_game)
        white_pos = white_game[random.randint(0, min(9, len(white_game)-1))]
        black_game = load_matrix(path_black_game)
        black_pos = black_game[random.randint(0, min(9, len(black_game)-1))]

        white_win_index = np.random.choice([0, 1])
        if white_win_index == 0:
            return white_pos, black_pos, white_win_index
        else:
            return black_pos, white_pos, white_win_index

    white_games = random.choices(os.listdir(parsed_path_white), k=epoch_size)
    black_games = random.choices(os.listdir(parsed_path_black), k=epoch_size)
    input_a = np.zeros((epoch_size,773), dtype=np.bool)
    input_b = np.zeros((epoch_size,773), dtype=np.bool)
    labels = np.zeros((epoch_size,2), dtype=np.bool)

    for i, (w,b) in enumerate(zip(white_games, black_games)):
        inp1, inp2, label_index = create_one_input(parsed_path_white+w, parsed_path_black+b)
        input_a[i] = inp1
        input_b[i] = inp2
        labels[i,label_index] = 1

    save_name = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase +
                                      string.digits) for _ in range(10))
    np.savez_compressed(save_dir+save_name, input_a, input_b, labels)


class DeepchessSequence(Sequence):
    def __init__(self, input_dir, batch_size, epoch_size):
        self.input_dir = input_dir
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.input_a, self.input_b, self.labels = self.read_input()

    def __len__(self):
        return math.ceil(self.epoch_size / self.batch_size)

    def __getitem__(self, idx):
        batch_input_a = self.input_a[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_input_b = self.input_b[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_labels = self.labels[idx*self.batch_size: (idx+1)*self.batch_size]
        return [batch_input_a, batch_input_b], batch_labels

    def on_epoch_end(self):
        self.input_a, self.input_b, self.labels = self.read_input()

    def read_input(self):
        inputs = os.listdir(self.input_dir)
        input = np.load(self.input_dir+random.choice(inputs))
        return input['arr_0'], input['arr_1'], input['arr_2']


class OnEpochEnd(tf.keras.callbacks.Callback):
  def __init__(self, callbacks):
    self.callbacks = callbacks

  def on_epoch_end(self, epoch, logs=None):
    for callback in self.callbacks:
      callback()
