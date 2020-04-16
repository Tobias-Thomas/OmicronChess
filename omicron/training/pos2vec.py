import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from omicron.util.db_preprocess import load_matrix
import numpy as np
import os
import random


def _init_autoencoder_model(layers=[773, 600, 400, 200, 100]):
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


def _create_train_examples(parsed_path_white, parsed_path_black,
                          samples=1e6, white_share=0.5):
    train_examples = np.zeros((samples,773))
    white_samples = int(samples * white_share)
    for i in range(white_samples):
        games = os.listdir(parsed_path_white)
        random_game = load_matrix(parsed_path_white+random.choice(games))
        train_examples[i] = random_game[random.randint(0,9)]
    for i in range(white_samples,samples):
        games = os.listdir(parsed_path_black)
        random_game = load_matrix(parsed_path_black+random.choice(games))
        train_examples[i] = random_game[random.randint(0,9)]
    return train_examples


def train_pos2vec_model(parsed_path_white, parsed_path_black, samples,
                        white_share=0.5, layers=[773, 600, 400, 200, 100]):
    model = _init_autoencoder_model(layers)
    train_examples = _create_train_examples(parsed_path_white, parsed_path_black,
                                            samples, white_share)
    with tf.device('/GPU:0'):
        history = model.fit(train_examples, train_examples, epochs=100, verbose=1)
    return model, history


def save_encoder(model, save_name, weight_dir, enc_layers=4):
    encoded = model.get_layer('encode%d'%(enc_layers-1)).output
    enc_model = Model(inputs=model.input, outputs=encoded)
    enc_model.save_weights(weight_dir+save_name)
