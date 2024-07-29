import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.backend import clear_session
import gc

from skopt import gp_minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from matplotlib import pyplot as plt

from pathlib import Path
import sys

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from base_optimizer import BaseOptimizer
from mathematical_models.f_on_f import FunctionOnFunctionModel
from mathematical_models.s_on_f import ScalarOnFunctionModel
from mathematical_models.s_on_s import ScalarOnScalar


class Autoencoder:
    def __init__(self, model, input_dim, latent_dim,
                 base=2, max_layers=None, alpha=0.0,
                 latent_space_activation='tanh', output_layer_activation='tanh'):
        self.model = model
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.base = base
        self.max_layers = max_layers
        self.alpha = alpha
        self.latent_space_activation = latent_space_activation
        self.output_space_activation = output_layer_activation

        self.encoder = None
        self.latent = None
        self.decoder = None
        self.autoencoder = None
        self.num_layers = None

        self.input_layer = None
        self.output_layer = None

        self.train_set = None
        self.val_set = None

    def _build_encoder(self):

        self.num_layers = int(np.log(self.input_dim / self.latent_dim)) / np.log(self.base)
        self.num_layers = min(self.num_layers, self.max_layers) if self.max_layers is not None else self.num_layers

        self.input_layer = Input(shape=(self.input_dim,))
        encoder = self.input_layer
        for layer in range(self.num_layers):
            n_neurons = int(self.input_dim / self.base ** (layer + 1))
            encoder = Dense(n_neurons, activation=LeakyReLU(alpha=self.alpha))(encoder)

        latent = Dense(self.latent_dim, activation=self.latent_space_activation, name='latent')(encoder)
        self.encoder = Model(self.input_layer, latent, name='encoder')

    def _build_decoder(self):

        latent_inputs = Input(shape=(self.latent_dim,))
        decoder = latent_inputs
        for layer in range(self.num_layers, 0, -1):
            n_neurons = int(self.input_dim / self.base ** layer)
            decoder = Dense(n_neurons, activation=LeakyReLU(alpha=self.alpha))(decoder)
        self.output_layer = Dense(self.input_dim, activation=self.output_space_activation)(decoder)
        self.decoder = Model(latent_inputs, self.output_layer, name='decoder')

    def _build_autoencoder(self):
        self._build_encoder()
        self._build_decoder()

        autoencoder_input = self.input_layer
        latent_representation = self.encoder(autoencoder_input)
        autoencoder_output = self.decoder(latent_representation)

        self.autoencoder = Model(autoencoder_input, autoencoder_output, name='autoencoder')

    def _get_custom_loss(self):
        if isinstance(self.model, ScalarOnFunctionModel):
            def custom_loss(y_pred):
                m = tf.shape(y_pred)[0]
                n = tf.shape(y_pred)[1]
                return self.model.compute_objective_tf(y_pred, m, n)
            return custom_loss
        elif isinstance(self.model, FunctionOnFunctionModel):
            return None
        elif isinstance(self.model, ScalarOnScalar):
            return None

    def compute_train_set(self, num_designs, runs,
                          epsilon=1e-10):
        design_matrix = []
        valid_count = 0

        for _ in range(10_000):
            if valid_count == num_designs:
                break
            candidate_matrix = np.random.uniform(-1, 1, size=(runs, self.model.Kx[0]))
            if self.model.J_cb is not None:
                Z = np.hstack((np.ones((runs, 1)), candidate_matrix @ self.model.J_cb))
            else:
                Z = np.hstack((np.ones((runs, 1)), candidate_matrix))
            ZtZ = Z.T @ Z
            determinant = np.linalg.det(ZtZ)
            if determinant > epsilon:
                design_matrix.append(candidate_matrix)
                valid_count += 1

        self.train_set, self.val_set = train_test_split(np.stack(design_matrix),
                                                        test_size=0.2,
                                                        random_state=42)
        pass

    def optimize(self, train_set, val_set, epochs, batch_size=32, patience=50,
                 optimizer=RMSprop(), loss='mse'):
        self._build_autoencoder()
        custom_loss = self._get_custom_loss()
        self.autoencoder.compile(optimizer=optimizer, loss=custom_loss)
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = self.autoencoder.fit(train_set, train_set,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       validation_data=(val_set, val_set),
                                       callbacks=[early_stopping])

        return self.autoencoder, self.encoder, self.decoder, history

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, latent):
        return self.decoder.predict(latent)


