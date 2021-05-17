import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Lambda, Dense, Dropout, Reshape, GlobalAvgPool2D
from keras.layers import ReLU, LeakyReLU, BatchNormalization, Input, Flatten, Concatenate
from tensorflow_datasets import load
from keras.datasets import mnist, cifar10
from keras import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from time import time
from utils import MinibatchDiscriminator

(X, Y), (X_test, Y_test) = mnist.load_data()
X = np.expand_dims(X, axis=-1)


class Dataset:
    def __init__(self, X, Y, n_classes=10):
        self.x = self.preprocess(X)
        self.y = Y
        self.n_classes = n_classes

    def get_sup_data(self, num_samples=100):
        x_sup, y_sup = [], []

        samples_per_class = int(num_samples / self.n_classes)
        for i in range(self.n_classes):

            idx = np.random.choice(len(self.x), samples_per_class, replace=False)
            x = self.x[idx]
            y = self.y[idx]
            x_sup.append(x)
            y_sup.append(y)
        x_sup = np.array(x_sup).reshape((-1, self.x.shape[1], self.x.shape[2], self.x.shape[3]))
        y_sup = np.array(y_sup).reshape((-1,))

        return x_sup, y_sup

    def get_real_data(self, num_samples):

        idx = np.random.choice(len(self.x), num_samples, replace=False)
        x_real = self.x[idx]
        return x_real

    @staticmethod
    def preprocess(X, type="tanh"):
        if type == 'tanh':
            X = (X - 127.5) / 127.5
            return X
        X = X / 255.0
        return X.astype(float)


in_shape = X[0].shape
n_classes = 10
latent_dim = 100
dataset = Dataset(X, Y, n_classes=n_classes)
sup, suplbl = dataset.get_sup_data()
real = dataset.get_real_data(len(dataset.x))
print(real.shape)


def get_discriminator(in_shape):
    # image input
    in_image = Input(shape=in_shape)
    # downsample
    fe = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    inter = Flatten()(fe)
    o = MinibatchDiscriminator(kernel_shape=(64, 50))(inter)
    # dropout
    fe = Dropout(0.4)(o)
    # output layer nodes
    out = Dense(n_classes)(fe)
    model = Model(in_image, [out, inter])
    return model


def get_generator(latent_dim):
    input_in = Input(shape=(latent_dim,))
    n_nodes = 256 * 7 * 7
    gen = Dense(n_nodes)(input_in)
    gen = ReLU()(gen)
    gen = Reshape((7, 7, 256))(gen)
    # upsample to 14x14
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = ReLU()(gen)
    # upsample to 28x28
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = ReLU()(gen)
    # output
    out_layer = Conv2D(1, (7, 7), activation='tanh', padding='same')(gen)
    model = Model(input_in, out_layer)
    return model


def disc_loss_sup(sup_logits, y_true):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, sup_logits)
    return loss


def disc_loss_unsup(unsup_logits, fake_logits):
    sig_pred = tf.reduce_sum(tf.exp(unsup_logits), axis=-1, keepdims=True) / (tf.reduce_sum(tf.exp(unsup_logits),
                                                                                            axis=-1,
                                                                                            keepdims=True
                                                                                            ) + 1)
    sig_pred_fake = tf.reduce_sum(tf.exp(fake_logits), axis=-1, keepdims=True) / (tf.reduce_sum(tf.exp(fake_logits),
                                                                                                axis=-1,
                                                                                                keepdims=True
                                                                                                ) + 1)
    loss_real = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(sig_pred), sig_pred)
    loss_fake = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(sig_pred_fake), sig_pred_fake)
    loss = loss_real + loss_fake
    return loss


def gen_loss(fake_logits):
    sig_pred_fake = tf.reduce_sum(tf.exp(fake_logits), axis=-1, keepdims=True) / (tf.reduce_sum(tf.exp(fake_logits),
                                                                                                axis=-1,
                                                                                                keepdims=True
                                                                                                ) + 1)
    loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(sig_pred_fake), sig_pred_fake)
    return loss


# feature matching loss for generator as in Salimans et. al 2014
def feature_matching_loss(int_real, int_fake):
    loss = tf.abs(tf.reduce_mean(tf.stop_gradient(int_real), axis=0) - tf.reduce_mean(int_fake, axis=0))
    loss = tf.reduce_mean(loss)
    return loss


disc = get_discriminator(in_shape=(28, 28, 1))
gen = get_generator(latent_dim=latent_dim)
disc_sup = get_discriminator(in_shape=(28, 28, 1))

g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)


ckpt_dir = 'ckpt-mini-batch-disc(1)'
ckpt = tf.train.Checkpoint(d_optimmizer=d_optimizer,
                           g_optimizer=g_optimizer,
                           discriminator=disc,
                           generator=gen,
                           )
manager = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=3)

if manager.latest_checkpoint:
    ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
    print(f"Restored !! {manager.latest_checkpoint}")

train_step_signature = [
    tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float64),
    tf.TensorSpec(shape=(None,), dtype=tf.uint8),
    tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.int32),
]

"""not a great method"""


@tf.function(input_signature=train_step_signature)  # to avoid retracing
def train_step(x_sup, y_sup, x_real, batch_size):
    z = tf.random.normal((batch_size, latent_dim))
    with tf.GradientTape(persistent=True) as tape:
        # train the discriminator
        fake_img = gen(z, training=True)

        sup_logits, _ = disc(x_sup, training=True)
        real_logits, int_real = disc(x_real, training=True)
        fake_logits, int_fake = disc(fake_img, training=True)

        unsup_loss = disc_loss_unsup(real_logits, fake_logits)
        sup_loss = disc_loss_sup(sup_logits, y_sup)

        g_loss = gen_loss(fake_logits)
        d_loss = sup_loss + unsup_loss

    gen_grads = tape.gradient(g_loss, gen.trainable_variables)
    disc_grads = tape.gradient(d_loss, disc.trainable_variables)
    g_optimizer.apply_gradients(zip(gen_grads, gen.trainable_variables))
    d_optimizer.apply_gradients(zip(disc_grads, disc.trainable_variables))

    return sup_loss, unsup_loss, g_loss


def generate_images(epoch):
    z_in = tf.random.normal((100, latent_dim))
    gen_imgs = gen(z_in, training=False)

    gen_imgs = (gen_imgs + 1) / 2

    for i in range(len(gen_imgs)):
        plt.subplot(10, 10, i+1)
        plt.axis("off")
        plt.imshow(gen_imgs[i, :, :, 0], cmap='gray_r')

    plt.savefig(f"image_at_ep{epoch+1}")
    plt.close()

    x_test = (X_test - 127.5) / 127.5

    label_logits, _ = disc.predict(x_test)
    real_labels = Y_test
    label_preds = tf.argmax(tf.nn.softmax(label_logits), axis=-1)
    accuracy = tf.metrics.Accuracy()
    accuracy.update_state(real_labels, label_preds)

    print(accuracy.result())


def train_loop(dataset, n_epochs=5, batch_size=100):

    x_sup, y_sup = dataset.get_sup_data()
    x_real = dataset.get_real_data(num_samples=len(dataset.x))
    bat_per_epoch = len(dataset.x) // batch_size
    n_steps = bat_per_epoch * n_epochs
    print(n_steps)

    # half_batch = batch_size // 2
    sup_data = tf.data.Dataset.from_tensor_slices((x_sup, y_sup)).shuffle(buffer_size=10000).repeat(600)\
        .batch(batch_size)
    unsup_data = tf.data.Dataset.from_tensor_slices(x_real).shuffle(buffer_size=10000).batch(batch_size)
    print("Starting the training loop....")
    for epoch in range(n_epochs):
        n = 0
        start = time()
        print(f"Epoch {epoch+1}/{n_epochs} : ")
        for (x_sup, y_sup), x_real in zip(sup_data, unsup_data):
            n = n+1
            sup_loss, unsup_loss, g_loss = train_step(x_sup, y_sup, x_real, batch_size)
            if n % 120 == 0:
                print('.', end='')
                print(f"un-sup_loss : {unsup_loss}, sup_loss : {sup_loss}, gen_loss : {g_loss}")

        if (epoch + 5) % 1 == 0:
            manager.save()
            generate_images(epoch)

        print(f"Time taken for epoch {epoch+1} : {time() - start}")


# train_loop(dataset)
generate_images(5)
