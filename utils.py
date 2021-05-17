import tensorflow as tf
from keras.layers import Layer


class MinibatchDiscriminator(Layer):
    def __init__(self, kernel_shape):
        super().__init__()
        self.kernel_shape = kernel_shape  # BxC in the paper

    def build(self, input_shape):
        self.batchT = self.add_weight(shape=(input_shape[-1], self.kernel_shape[0], self.kernel_shape[1]),
                                      initializer=tf.random_normal_initializer(stddev=0.05),
                                      trainable=True,
                                      name="kernel"
                                      )

    def call(self, inputs, **kwargs):
        if tf.rank(inputs) > 2:
            inputs = tf.keras.layers.Flatten()(inputs)

        M = tf.einsum("ij,jkl->ikl", inputs, self.batchT)  # nxA to nxBXC as in the paper
        row_l1_dist = tf.abs(tf.expand_dims(M, axis=1) - M)
        c = tf.exp(-tf.reduce_sum(row_l1_dist, axis=3))
        out = tf.reduce_sum(c, axis=1)

        return tf.concat([inputs, out], axis=-1)


if __name__ == "__main__":
    B = 100
    C = 50
    mini_batch_discrimination = MinibatchDiscriminator(kernel_shape=(B, C))




