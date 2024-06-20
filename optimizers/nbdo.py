import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

from pathlib import Path
import sys

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from base_optimizer import BaseOptimizer
from mathematical_models.f_on_f import FunctionOnFunctionModel
from mathematical_models.s_on_f import ScalarOnFunctionModel


class NBDO(BaseOptimizer):
    def __init__(self, model):
        super().__init__(model)

    def create_ae(self, in_dim, lat_dim, lat_act='tanh',
                  out_act='tanh', max_layers=None, alpha=0., base=2):

        # settings and checks
        num_layers = int(np.log(in_dim) / np.log(base))
        if max_layers is not None:
            num_layers = min(num_layers, max_layers)

        # Input
        input_layer = Input(shape=(in_dim,))
        # Encoder
        encoder = input_layer
        for i in range(num_layers):
            n_neurons = int(in_dim / (2 ** (i + 1)))
            encoder = Dense(n_neurons, activation=LeakyReLU(alpha=alpha))(encoder)
        # Latent space
        latent_space = Dense(lat_dim, activation=lat_act, name='lat_space')(encoder)
        # Decoder
        decoder = latent_space
        for i in range(num_layers, 0, -1):
            n_neurons = int(in_dim / (2 ** i))
            decoder = Dense(n_neurons, activation=LeakyReLU(alpha=alpha))(decoder)
        # Output
        output_layer = Dense(in_dim, activation=out_act)(decoder)

        # Initiate model
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        encoder = Model(inputs=input_layer, outputs=latent_space)
        encoder_input = Input(shape=(lat_dim,))

        decoder_output = encoder_input
        decoder_layers = autoencoder.layers[-(num_layers + 1):]
        for layer in decoder_layers:
            decoder_output = layer(decoder_output)
        decoder = Model(inputs=encoder_input, outputs=decoder_output)

        return autoencoder, encoder, decoder

    def optimize(self, autoencoder, encoder, decoder, train_data, val_data, epochs=1_000,
                 batch_size=32, patience=50, optimizer=RMSprop, loss=tf.keras.losses.Huber(), monitor='val_loss',
                 alpha=1., m=None, n=None, J_cb=None, noise=0, optimizer_kwargs=None,SEED=42):

        def objective_function_tf(X, m, n, J_cb=None, noise=0):
            batch_size = tf.shape(X)[0]
            ones = tf.ones((batch_size, m, 1))
            X = tf.reshape(X, (-1, m, n))
            Z = tf.concat([ones, tf.matmul(X, J_cb)], axis=2)

            Z_transpose_Z = tf.matmul(Z, Z, transpose_a=True)
            det_Z_transpose_Z = tf.linalg.det(Z_transpose_Z)
            epsilon = 1e-06
            condition = tf.abs(det_Z_transpose_Z)[:, None, None] < epsilon

            identity_matrix = tf.eye(tf.shape(Z_transpose_Z)[1], tf.shape(Z_transpose_Z)[2])
            diagonal_part = tf.linalg.diag_part(Z_transpose_Z) + epsilon
            Z_transpose_Z_epsilon = Z_transpose_Z + tf.linalg.diag(diagonal_part - tf.linalg.diag_part(Z_transpose_Z))
            regularized_matrix = tf.where(condition, Z_transpose_Z_epsilon, Z_transpose_Z)

            M = tf.linalg.inv(regularized_matrix)
            result = tf.linalg.trace(M) + tf.random.normal([], mean=0, stddev=noise)
            result = tf.where(result < 0, tf.constant(1e10), result)
            return tf.reduce_mean(result)

        def combined_loss(alpha, loss_function, m, n, J_cb=None, noise=0):
            def custom_loss(y_true, y_pred):
                reconstruction_loss = loss_function(y_true, y_pred)
                objective_value = objective_function_tf(y_pred, m, n, J_cb=J_cb, noise=noise)
                return (1 - alpha) * reconstruction_loss + alpha * objective_value

            return custom_loss

        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        # Create a new optimizer instance
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer = optimizer(**optimizer_kwargs)

        custom_loss = combined_loss(alpha, loss, m, n, J_cb=J_cb, noise=noise)

        # Train AE with EarlyStopping
        autoencoder.compile(optimizer=optimizer, loss=custom_loss)
        early_stopping = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
        history = autoencoder.fit(train_data, train_data,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  validation_data=(val_data, val_data),
                                  callbacks=[early_stopping])

        return autoencoder, encoder, decoder, history