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

class Pix2Pix:   
    def __init__(self, g=None, d=None, shape = (None, None, 1), verbose=0):
        self.verbose = verbose
        self.shape = shape

        if g is None or d is None:
            raise Exception('g or d cannot be None and must be a `tf.keras.Model`')
        self.d = d
        self.g = g
        self.patch_gan_hw = patch_gan_hw

    def compile(self, optimizer=None, metrics=[], d_loss='mse', g_loss = ['mse', 'mae'], loss_weights = [1, 100]):
        if optimizer is None:
            raise Exception('optimizer cannot be None')
        
        self.optimizer = optimizer
        self.metrics = metrics
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.loss_weights = loss_weights
        
        self.d.compile(loss=self.d_loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.d.trainable = False

        img_A = tf.keras.layers.Input(shape=self.shape)   # Input images from both domains
        img_B = tf.keras.layers.Input(shape=self.shape)
        
        fake_B = self.g(img_A)                            # Translate images to the other domain
        valid = self.d(tf.keras.layers.Concatenate(axis=-1)([img_A, fake_B]))
        
        self.combined = tf.keras.Model(inputs=[img_A, img_B], outputs=[valid, fake_B])
        
        # Combined model trains generators to fool discriminators
        self.combined.compile(loss=g_loss,
                              loss_weights=loss_weights,
                              optimizer=optimizer)
        
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
            metrics.extend(['d_acc', 'd_loss', 'g_loss'])
         
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
                
                imgs_A, imgs_B = self.sess.run(self.dataset_next)
                
                self.patch_gan_size = np.concatenate([[imgs_A.shape[0]], self.d.output_shape[1:]])

                # Translate images to opposite domain
                fake_B = self.g.predict(imgs_A)

                # Train the discriminators (original images = real / translated = Fake)
                d_loss_real = self.d.train_on_batch(np.concatenate([imgs_A, imgs_B], axis=-1), np.ones(self.patch_gan_size))
                d_loss_fake = self.d.train_on_batch(np.concatenate([imgs_A, fake_B], axis=-1), np.zeros(self.patch_gan_size))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [np.ones(self.patch_gan_size), imgs_B])
                
                self.log['d_loss'] = d_loss[0]
                self.log['d_acc'] = 100*d_loss[1]
                self.log['g_loss'] = g_loss[0]
                                
                for callback in callbacks: callback.on_batch_end(step, logs=self.log)
            
            if validation_data is not None:
                forward_metrics = self.validate(validation_steps)
                for key, value in forward_metrics.items():
                    self.log['val_' + key] = value
            
            for callback in callbacks: callback.on_epoch_end(epoch, logs=self.log)
        for callback in callbacks: callback.on_train_end(logs=self.log)
