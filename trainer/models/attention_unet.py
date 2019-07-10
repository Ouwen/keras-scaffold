import tensorflow as tf
from trainer.models.model_base import ModelBase

def AttentionBlock(signal, gate, i_filters):   
    g1 = tf.keras.layers.Conv2D(i_filters, 1)(gate) # small
    g1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(g1)
    x1 = tf.keras.layers.Conv2D(i_filters, 1)(signal) # big
    g1_x1 = tf.keras.layers.Add()([g1, x1])
    psi = tf.keras.layers.Activation(activation=tf.nn.relu)(g1_x1)
    psi = tf.keras.layers.Conv2D(1,1)(psi)
    psi = tf.keras.layers.Activation(activation=tf.nn.sigmoid)(psi)
    
    return tf.keras.layers.Multiply()([signal, psi])

class AttentionUnetModel(ModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            
    def __call__(self):
        downsample_path = []
        inputs = tf.keras.layers.Input(shape=self.shape)
        x = inputs
        
        for idx, filter_num in enumerate(self.filters):
            x = self.Conv2D(filter_num)(x)
            x = self.Activation(x)
            x = self.Dropout()(x)
            x = self.Conv2D(filter_num)(x)
            x = self.Activation(x)
            x = self.Dropout()(x)
            if idx != len(self.filters)-1:
                downsample_path.append(x)
                x = tf.keras.layers.MaxPool2D(padding=self.padding)(x)

        downsample_path.reverse()
        reverse_filters = list(self.filters[:-1])
        reverse_filters.reverse()

        # Upsampling path
        for idx, filter_num in enumerate(reverse_filters):
            x = tf.keras.layers.concatenate([self.Upsample(filter_num)(x), 
                                             AttentionBlock(downsample_path[idx], x, filter_num)])
            x = self.Conv2D(filter_num)(x)
            x = self.Activation(x)
            x = self.Dropout()(x)
            x = self.Conv2D(filter_num)(x)
            x = self.Activation(x)
            x = self.Dropout()(x)

        x = tf.keras.layers.Conv2D(1, 1)(x)

        return tf.keras.Model(inputs, x)