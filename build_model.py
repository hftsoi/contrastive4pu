from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from utils import *

input_shape = (64, 50, 1)
embedding_dim = 128
projection_dim = 64
c_inv = 25
c_var = 25
c_cov = 1

def build_encoder(input_shape=input_shape, embedding_dim=embedding_dim):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(embedding_dim)(x)
    outputs = tf.keras.layers.LayerNormalization()(x)

    return tf.keras.Model(inputs, outputs, name="encoder")


def build_projection_head(embedding_dim=embedding_dim, projection_dim=projection_dim):
    inputs = tf.keras.Input(shape=(embedding_dim,))
    x = tf.keras.layers.Dense(embedding_dim, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(projection_dim)(x)

    return tf.keras.Model(inputs, outputs, name="projection_head")


def vicreg_loss(z1, z2, c_inv=c_inv, c_var=c_var, c_cov=c_cov, epsilon=0.0001):
    # invariance between positive views
    loss_inv = tf.reduce_mean(tf.square(z1 - z2))

    # maximize variance per feature dim across sample batch (avoid collapse--learning constant vector)
    std_z1 = tf.sqrt(tf.math.reduce_variance(z1, axis=0) + epsilon)
    std_z2 = tf.sqrt(tf.math.reduce_variance(z2, axis=0) + epsilon)
    loss_var = tf.reduce_mean(tf.nn.relu(1 - std_z1)) + tf.reduce_mean(tf.nn.relu(1 - std_z2))

    # minimize covariance between feature dims (to reduce learning feature redundancy)
    z1_centered = z1 - tf.reduce_mean(z1, axis=0)
    z2_centered = z2 - tf.reduce_mean(z2, axis=0)
    batch_size = tf.cast(tf.shape(z1)[0], tf.float32)
    # covariance matrices for z1 z2
    cov_z1 = tf.matmul(tf.transpose(z1_centered), z1_centered) / (batch_size - 1)
    cov_z2 = tf.matmul(tf.transpose(z2_centered), z2_centered) / (batch_size - 1)
    # get diagonal parts
    diag_z1 = tf.linalg.diag(tf.linalg.diag_part(cov_z1))
    diag_z2 = tf.linalg.diag(tf.linalg.diag_part(cov_z2))
    # subtract diagonal parts from cov matrices to get off-diagonal parts (cov between features)
    loss_cov_z1 = tf.reduce_sum(tf.square(cov_z1 - diag_z1)) / tf.cast(tf.shape(z1)[1], tf.float32)
    loss_cov_z2 = tf.reduce_sum(tf.square(cov_z2 - diag_z2)) / tf.cast(tf.shape(z2)[1], tf.float32)
    loss_cov = loss_cov_z1 + loss_cov_z2

    loss_inv *= c_inv
    loss_var *= c_var
    loss_cov *= c_cov

    loss = loss_inv + loss_var + loss_cov

    return loss, loss_inv, loss_var, loss_cov


class VICRegModel(tf.keras.Model):
    def __init__(self, encoder, projection_head, c_inv=c_inv, c_var=c_var, c_cov=c_cov, **kwargs):
        super(VICRegModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.projection_head = projection_head
        self.c_inv = c_inv
        self.c_var = c_var
        self.c_cov = c_cov

    def compile(self, optimizer, **kwargs):
        super(VICRegModel, self).compile(**kwargs)
        self.optimizer = optimizer

    def train_step(self, data):
        # view1, view2
        x1, x2 = data
        with tf.GradientTape() as tape:
            emb1 = self.encoder(x1, training=True)
            emb2 = self.encoder(x2, training=True)

            z1 = self.projection_head(emb1, training=True)
            z2 = self.projection_head(emb2, training=True)

            loss, loss_inv, loss_var, loss_cov = vicreg_loss(z1, z2,
                                                             c_inv=self.c_inv,
                                                             c_var=self.c_var,
                                                             c_cov=self.c_cov)
        
        vars = self.encoder.trainable_variables + self.projection_head.trainable_variables
        grads = tape.gradient(loss, vars)
        self.optimizer.apply_gradients(zip(grads, vars))

        return {"loss": loss, "loss_inv": loss_inv, "loss_var": loss_var, "loss_cov": loss_cov}


def build_embedding_classifier(encoder, input_shape=input_shape, encoder_trainable=False):
    # update or freeze the encoder weights
    encoder.trainable = encoder_trainable

    inputs = tf.keras.Input(shape=input_shape)
    # option-training here concerns about training/inference mode in dropout, batchnorm etc. 
    x = encoder(inputs, training=False)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs, outputs, name="embedding_classifier")


class FreezeEncoderCallback(tf.keras.callbacks.Callback):
    def __init__(self, freeze_epoch):
        super(FreezeEncoderCallback, self).__init__()
        self.freeze_epoch = freeze_epoch

    def on_epoch_end(self, epoch):
        if epoch == self.freeze_epoch - 1:
            self.model.get_layer('encoder').trainable = False
            self.model.compile(optimizer=self.model.optimizer,
                               loss=self.model.loss,
                               metrics=self.model.metrics)
            print(f"encoder frozen starting at epoch {self.freeze_epoch}")
            

def build_standalone_classifier(input_shape=input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs, outputs, name="standalone_classifier")

