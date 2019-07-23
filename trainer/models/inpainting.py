# Original work Copyright (c) 2019 Ouwen Huang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Flatten, LeakyReLU, Dense

def resize(x, scale=2, to_shape=None, align_corners=True, func=tf.image.resize_nearest_neighbor):
    if to_shape is None:
        x_shape = tf.cast(tf.shape(x), dtype=tf.float32)
        new_xs = [tf.cast(x_shape[1]*scale, dtype=tf.int32), tf.cast(x_shape[2]*scale, dtype=tf.int32)]
        return func(x, new_xs, align_corners=align_corners)
    else:
        return func(x, [to_shape[0], to_shape[1]], align_corners=align_corners)

def set_shape_like(inputs):
    x = inputs[0]
    template = inputs[1]
    x.set_shape(template.get_shape().as_list())
    return x
    
class InPainting:
    def __init__(self, shape=(None, None, 1), base_filters=32):
        self.base_filters = base_filters
        self.shape = shape
        
    def __call__(self):
        original_inputs = tf.keras.layers.Input(shape=self.shape)
        mask = tf.keras.layers.Input(shape=self.shape) # 1 where missing
        mask_s = tf.keras.layers.Lambda(lambda x: resize(x[0], scale=x[1]))((mask, 0.25))
        inputs = tf.keras.layers.Lambda(lambda x: x[1]*tf.math.abs(1.0 - x[0]))((mask, original_inputs)) # mask the input image

        # stage1, coarse network generation
        x = Concatenate(axis=-1, name='stage1')([inputs, mask])

        x = Conv2D(self.base_filters, (5,5), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=2, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=2, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(2,2), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(4,4), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(8,8), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(16,16), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(tf.keras.layers.UpSampling2D()(x))
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(tf.keras.layers.UpSampling2D()(x))
        x = Conv2D(self.base_filters//2, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(1, (3,3), strides=1, padding='same')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1.0, 1.0))(x)
        x_stage1 = x

        # reset image pixels outside of mask
        x = tf.keras.layers.Lambda(lambda x: x[0]*x[1] + x[2]*tf.math.abs(1.0 - x[1]))((x, mask, inputs))
        
        # stage2 fine network generation
        xnow = Concatenate(axis=-1, name='stage2')([x, mask])
      
        ## conv branch
        x = Conv2D(self.base_filters, (5,5), strides=1, padding='same', activation=tf.nn.elu)(xnow)
        x = Conv2D(self.base_filters, (3,3), strides=2, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=2, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        temp = x
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(2,2), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(4,4), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(8,8), padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), dilation_rate=(16,16), padding='same', activation=tf.nn.elu)(x)
        x = tf.keras.layers.Lambda(set_shape_like)((x, temp)) # Workaround for the following bug: #
        
        x_hallu = x

        ## attention branch
        x = Conv2D(self.base_filters, (5,5), strides=1, padding='same', activation=tf.nn.elu)(xnow)
        x = Conv2D(self.base_filters, (3,3), strides=2, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=2, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.relu)(x)
        x = tf.keras.layers.Lambda(lambda x: contextual_attention(x[0], x[0], masks=x[1], ksize=3, stride=1, rate=2), name='attention')((x, mask_s))
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        pm = x

        ## combine branches and upsample
        x = Concatenate(axis=-1)([x_hallu, pm])

        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(4*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(tf.keras.layers.UpSampling2D()(x))
        x = Conv2D(2*self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(self.base_filters, (3,3), strides=1, padding='same', activation=tf.nn.elu)(tf.keras.layers.UpSampling2D()(x))
        x = Conv2D(self.base_filters//2, (3,3), strides=1, padding='same', activation=tf.nn.elu)(x)
        x = Conv2D(1, (3,3), strides=1, padding='same')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1.0, 1.0))(x)
        x = tf.keras.layers.Lambda(lambda x: x[0]*x[1] + x[2]*tf.math.abs(1.0 - x[1]))((x, mask, inputs))

        return tf.keras.Model([original_inputs, mask], x)

    
def wgan_local_discriminator(base_filters=64, shape=(None, None, 1)):
    inputs = tf.keras.layers.Input(shape=shape)
    mask = tf.keras.layers.Input(shape=shape) # 1 where missing

    x = Conv2D(base_filters, (5,5), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(base_filters*2, (5,5), strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(base_filters*4, (5,5), strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(base_filters*8, (5,5), strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(1)(x)

    return tf.keras.Model([inputs, mask], x)

def wgan_global_discriminator(base_filters=64, shape=(None, None, 1)):
    inputs = tf.keras.layers.Input(shape=shape)
    mask = tf.keras.layers.Input(shape=shape) # 1 where missing

    x = Conv2D(base_filters, (5,5), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(base_filters*2, (5,5), strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(base_filters*4, (5,5), strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(base_filters*4, (5,5), strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(1)(x)

    return tf.keras.Model([inputs, mask], x)


def contextual_attention(f, b, masks=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    raw_int_bs[0] = tf.shape(b)[0]
    
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    int_fs[0] = tf.shape(f)[0]
    f_groups = tf.expand_dims(f, axis=1)
    
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    
    # process mask
    if masks is None:
        masks = tf.zeros([bs[0], bs[1], bs[2], 1])
    else:
        masks = resize(masks, scale=1./rate, func=tf.image.resize_nearest_neighbor)
        
    w_groups = w
    raw_w_groups = raw_w
    fuse_weight = tf.reshape(tf.eye(fuse_k), [fuse_k, fuse_k, 1, 1])

    def per_instance_func(groups):
        xi = groups[0]
        wi = tf.expand_dims(groups[1], axis=0)
        raw_wi = tf.expand_dims(groups[2], axis=0)
        mask = tf.expand_dims(groups[3], axis=0)
        
        m = tf.extract_image_patches(mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
        m = tf.reshape(m, [1, -1, ksize, ksize, 1])
        m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
        m = m[0]
        mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)

        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*softmax_scale, 3)
        yi *=  mm  # mask

        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        return tf.squeeze(yi)
    
    y = tf.map_fn(per_instance_func, (f_groups, w_groups, raw_w_groups, masks), dtype=tf.float32)
    y.set_shape(raw_int_fs)
    
    return y
