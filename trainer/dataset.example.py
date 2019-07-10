import tensorflow as tf
import numpy as np
import pandas as pd
from trainer.get_default import config

class Dataset():
    def __init__(self, image_dir=None, bucket_dir=config.bucket_dir, shape=(None, None, 1)):
        self.shape = shape
        self.image_dir = image_dir
        self.bucket_dir = bucket_dir
        if self.image_dir is None: self.image_dir = bucket_dir
        
    def read_file(self, shape=None):
        def read_file_op(filenames, shape=shape):
            a = tf.io.read_file(filenames[0])
            b = tf.io.read_file(filenames[1])
            a = tf.decode_raw(input_bytes = a, out_type=tf.float32, little_endian=True)
            b  = tf.decode_raw(input_bytes = b, out_type=tf.float32, little_endian=True)
            a = tf.reshape(a, (2048, 2048, 1))
            b = tf.reshape(b, (2048, 2048, 1))
            
            # normalize 0 to 1
            a = (a - tf.math.reduce_min(a))/(tf.math.reduce_max(a) - tf.math.reduce_min(a))
            b = (b - tf.math.reduce_min(b))/(tf.math.reduce_max(b) - tf.math.reduce_min(b))
            
            # distort
            ds = tf.random_uniform([], minval=tf.to_int32(0), maxval=tf.to_int32(700), dtype=tf.int32)
            c = tf.concat([a, b], axis=-1)
            c = tf.image.resize(c, [2048 - ds, 2048 - ds])
            
            # crop
            crop_shape = list(shape)
            crop_shape[-1] = 2
            cropped = tf.image.random_crop(both, crop_shape)
            a, b = tf.split(cropped, 2, axis=-1)
            
            return a, b
        return read_file_op

    def get_dataset(self, csv, shape=None, batch_size=4):
        df = pd.read_csv(tf.io.gfile.GFile(csv, 'rb'))
        df['a'] = df['a'].apply(lambda x: '{}/{}'.format(self.image_dir, x))
        df['b'] = df['b'].apply(lambda x: '{}/{}'.format(self.image_dir, x))
        filelist = list(zip(df.a, df.b))
        count = len(filelist)
        dataset = tf.data.Dataset.from_tensor_slices(filelist)
        dataset = dataset.shuffle(count).repeat()
        
        dataset = dataset.map(self.read_file(shape=shape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device('gpu:0'))
        return dataset, count
    

def make_shape(image, shape=None, seed=0, divisible=16):
    np.random.seed(seed=seed)
    image_height = image.shape[0]
    image_width = image.shape[1]

    shape = shape if shape is not None else image.shape
    height = shape[0] if shape[0] % divisible == 0 else (divisible - shape[0] % divisible) + shape[0]
    width = shape[1] if shape[1] % divisible == 0 else (divisible - shape[1] % divisible) + shape[1]

    # Pad data to batch height and width with reflections, and randomly crop
    if image_height < height:
        remainder = height - image_height
        if remainder % 2 == 0:
            image = np.pad(image, ((int(remainder/2), int(remainder/2)), (0,0)), 'reflect')
        else:
            remainder = remainder - 1
            image = np.pad(image, ((int(remainder/2) + 1, int(remainder/2)), (0,0)), 'reflect')
    elif image_height > height:
        start = np.random.randint(0, image_height - height)
        image = image[start:start+height, :]

    if image_width < width:
        remainder = width - image_width
        if remainder % 2 == 0:
            image = np.pad(image, ((0,0), (int(remainder/2), int(remainder/2))), 'reflect')
        else:
            remainder = remainder - 1
            image = np.pad(image, ((0,0), (int(remainder/2) + 1, int(remainder/2))), 'reflect')
    elif image_width > width:
        start = np.random.randint(0, image_width - width)
        image = image[:, start:start+width]
    return image, (image_height, image_width)
