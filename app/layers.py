import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Conv1D, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization,
)


class AttentionLayer(Layer):
    """Bahdanau-style attention — kept for backward compatibility."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_W', shape=(input_shape[-1], input_shape[-1]),
                                  initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_b', shape=(input_shape[-1],),
                                  initializer='zeros', trainable=True)
        self.u = self.add_weight(name='att_u', shape=(input_shape[-1],),
                                  initializer='glorot_uniform', trainable=True)

    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        return tf.reduce_sum(x * tf.expand_dims(weights, -1), axis=1)

    def get_config(self):
        return super().get_config()


class SEBlock(Layer):
    """Squeeze-and-Excitation block for TCN."""
    def __init__(self, ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        ch = input_shape[-1]
        self.squeeze = GlobalAveragePooling1D()
        self.fc1 = Dense(ch // self.ratio, activation='relu')
        self.fc2 = Dense(ch, activation='sigmoid')

    def call(self, x):
        se = self.squeeze(x)
        se = self.fc1(se)
        se = self.fc2(se)
        se = tf.expand_dims(se, 1)
        return x * se

    def get_config(self):
        config = super().get_config()
        config["ratio"] = self.ratio
        return config


class TCNBlock(Layer):
    """Temporal Convolutional Block with dilated causal convolutions."""
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.conv1 = Conv1D(self.filters, self.kernel_size, dilation_rate=self.dilation_rate,
                            padding='causal', activation=None)
        self.bn1 = LayerNormalization()
        self.conv2 = Conv1D(self.filters, self.kernel_size, dilation_rate=self.dilation_rate,
                            padding='causal', activation=None)
        self.bn2 = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate)
        self.se = SEBlock(ratio=4)
        if input_shape[-1] != self.filters:
            self.residual_conv = Conv1D(self.filters, 1, padding='same')
        else:
            self.residual_conv = None

    def call(self, x, training=None):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = tf.nn.gelu(out)
        out = self.dropout(out, training=training)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.residual_conv is not None:
            res = self.residual_conv(res)
        out = tf.nn.gelu(out + res)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "dropout_rate": self.dropout_rate,
        })
        return config


CUSTOM_OBJECTS = {
    "AttentionLayer": AttentionLayer,
    "SEBlock": SEBlock,
    "TCNBlock": TCNBlock,
}
