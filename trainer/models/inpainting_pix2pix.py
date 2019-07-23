# Original work Copyright (c) 2017 Erik Linder-Norén
# Modified work Copyright 2019 Ouwen Huang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

import tensorflow as tf
import numpy as np
from functools import partial

def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)

def random_interpolates(inputs):
    x = inputs[0]
    y = inputs[1]
    
    shape = tf.shape(x)
    x = tf.reshape(x, [shape[0], -1])
    y = tf.reshape(y, [shape[0], -1])
    alpha = tf.random_uniform(shape=[shape[0], 1])
    interpolates = x + alpha*(y - x)
    return tf.reshape(interpolates, shape)

def gradient_penalty_loss(y_true, y_pred, averaged_samples=None, mask=None, norm=1.0):
    gradients = tf.gradients(y_pred, averaged_samples)[0]
    if mask is None: 
        mask = tf.ones_like(gradients)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients) * mask, axis=[1, 2, 3]))
    return tf.reduce_mean(tf.square(slopes - norm))

class InPaintingPix2Pix:   
    def __init__(self, g=None, dl=None, dg=None, shape = (None, None, 1), verbose=0):
        self.verbose = verbose
        self.shape = shape

        if g is None or dl is None or dg is None:
            raise Exception('g, dl, or dg cannot be None and must be a `tf.keras.Model`')
        self.dl = dl
        self.dg = dg
        self.g = g
    
    def compile(self, optimizer=None, metrics=[]):
        if optimizer is None:
            raise Exception('optimizer cannot be None')
    
        self.optimizer = optimizer
        self.metrics = metrics
        
        # Inputs
        real_image = tf.keras.layers.Input(shape=self.shape)   # Input images from both domains
        mask = tf.keras.layers.Input(shape=self.shape)
        
        # Build the discriminators
        self.g.trainable = False

        # Build global discriminator
        fake_image = self.g([real_image, mask])
        global_interpolated_img = tf.keras.layers.Lambda(random_interpolates)((real_image, fake_image))
        global_fake = self.dg([fake_image, mask])
        global_valid = self.dg([real_image, mask])
        global_validity_interpolated = self.dg([global_interpolated_img, mask])
        global_partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=global_interpolated_img, mask=mask)
        global_partial_gp_loss.__name__ = 'global_gradient_penalty'
        
        self.global_critic_model = tf.keras.Model(inputs=[real_image, mask],
                                                  outputs=[global_valid, 
                                                           global_fake, 
                                                           global_validity_interpolated])
        
        self.global_critic_model.compile(loss=[wasserstein_loss,
                                               wasserstein_loss,
                                               global_partial_gp_loss],
                                         optimizer=optimizer,
                                         loss_weights=[1, 1, 10])
        
        # Build local discriminator
        fake_image = self.g([real_image, mask])
        local_interpolated_img = tf.keras.layers.Lambda(random_interpolates)((real_image, fake_image))
        local_fake = self.dl([fake_image, mask])
        local_valid = self.dl([real_image, mask])
        local_validity_interpolated = self.dl([local_interpolated_img, mask])
        local_partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=local_interpolated_img, mask=mask)
        local_partial_gp_loss.__name__ = 'local_gradient_penalty'
        
        self.local_critic_model = tf.keras.Model(inputs=[real_image, mask],
                                           outputs=[local_valid, 
                                                    local_fake, 
                                                    local_validity_interpolated])
        
        self.local_critic_model.compile(loss=[wasserstein_loss,
                                              wasserstein_loss,
                                              local_partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        
        # Build the Generator
        self.local_critic_model.trainable = False
        self.global_critic_model.trainable = False
        self.g.trainable = True
        fake_image = self.g([real_image, mask])
        local_valid = self.dl([fake_image, mask])
        global_valid = self.dg([fake_image, mask])
        self.combined_generator_model = tf.keras.Model([real_image, mask], [local_valid, global_valid, fake_image])
        self.combined_generator_model.compile(optimizer=optimizer, 
                                              loss=[wasserstein_loss, wasserstein_loss, 'mae'])
        
        ## TODO deprecated
        self.metrics_graph = tf.Graph()
        with self.metrics_graph.as_default():
            self.val_batch_placeholder = tf.placeholder(tf.float32)
            self.val_fake_placeholder = tf.placeholder(tf.float32)
            self.output_metrics = {}
            for metric in self.metrics:
                self.output_metrics[metric.__name__] = metric(self.val_batch_placeholder, self.val_fake_placeholder)
        self.metrics_session = tf.Session(graph=self.metrics_graph)
    
    def validate(self, validation_steps):
        metrics_summary = {}
        for metric in self.metrics:
            metrics_summary[metric.__name__] = []
        
        for step in range(validation_steps):
            val_batch = self.sess.run(self.dataset_val_next)
            B_batch = val_batch[1]
            fake_B = self.g.predict(val_batch[0])
                                    
            forward_metrics = self.metrics_session.run(self.output_metrics, feed_dict={
                self.val_batch_placeholder: B_batch,
                self.val_fake_placeholder: fake_B
            })
            
            for key, value in forward_metrics.items():
                if key not in metrics_summary:
                    metrics_summary[key] = []
                metrics_summary[key].append(value)
        
        # average all metrics
        for key, value in metrics_summary.items():
            metrics_summary[key] = np.mean(value)
        return metrics_summary
    
    def fit(self, dataset, batch_size=8, steps_per_epoch=10, epochs=3, validation_data=None, validation_steps=10, 
            callbacks=[]):
        
        if not hasattr(self, 'sess'):
            self.dataset_next = dataset.make_one_shot_iterator().get_next()
            self.dataset_val_next = validation_data.make_one_shot_iterator().get_next()
            
            metrics = ['val_' + metric.__name__ for metric in self.metrics]
            metrics.extend(['d_loss', 'g_loss', 'mse', 'mae'])
         
            self.sess = tf.Session()
            for callback in callbacks: 
                callback.set_model(self.g)
                callback.set_params({
                    'verbose': self.verbose,
                    'epochs': epochs,
                    'steps': steps_per_epoch,
                    'metrics': metrics
                })
                
        self.log = {
            'size': batch_size
        }
        
        for callback in callbacks: callback.on_train_begin(logs=self.log)
        for epoch in range(epochs):
            for callback in callbacks: callback.on_epoch_begin(epoch, logs=self.log)
            for step in range(steps_per_epoch):
                for callback in callbacks: callback.on_batch_begin(step, logs=self.log)
                
                image, mask = self.sess.run(self.dataset_next)
                
                d_local_loss = self.local_critic_model.train_on_batch([image, mask], 
                                                                      [np.ones((image.shape[0],1)), 
                                                                       -1*np.ones((image.shape[0],1)),
                                                                       np.zeros((image.shape[0],1))])
                
                d_global_loss = self.global_critic_model.train_on_batch([image, mask], 
                                                                        [np.ones((image.shape[0],1)), 
                                                                         -1*np.ones((image.shape[0],1)),
                                                                         np.zeros((image.shape[0],1))])
                
                g_loss = self.global_critic_model.train_on_batch([image, mask],
                                                                 [np.ones((image.shape[0],1)), 
                                                                  np.ones((image.shape[0],1)),
                                                                  image])
                
                fake_image = self.g.predict([image, mask])
                
                self.log['d_loss'] = 0.5*(d_local_loss[0]+d_local_loss[1] + d_global_loss[0]+d_global_loss[1])
                self.log['g_loss'] = g_loss[0]+g_loss[1]
                self.log['mse'] = (np.square(fake_image - image)).mean(axis=None)
                self.log['mae'] = g_loss[2]
                
                for callback in callbacks: callback.on_batch_end(step, logs=self.log)
            
            if validation_data is not None:
                forward_metrics = self.validate(validation_steps)
                for key, value in forward_metrics.items():
                    self.log['val_' + key] = value
            
            for callback in callbacks: callback.on_epoch_end(epoch, logs=self.log)
        for callback in callbacks: callback.on_train_end(logs=self.log)
