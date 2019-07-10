import tarfile
import tensorflow as tf
import argparse
import os
import subprocess
from trainer import config

class DownloadData(tf.keras.callbacks.Callback):
    def __init__(self, image_dir='/tmp/my-data', bucket_dir=config.bucket_dir, force=False):
        super()
        self.image_dir = image_dir
        self.bucket_dir = bucket_dir
        self.force = force
    
    def download_data(self, directory, bucket_dir, force=False):
        if not os.path.exists(directory) or force:
            if not os.path.exists(directory): os.mkdir(directory)
            subprocess.call(['df'])
            subprocess.call('gsutil -m cp -r {} {}'.format(bucket_dir, directory).split(' '))

    def on_train_begin(self, logs={}):
        self.download_data(self.image_dir, self.bucket_dir, force=self.force)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_dir', help='Cloud image directory location')
    parser.add_argument('--image_dir',  help='Local image directory location')
    parser.add_argument('--force', action='store_true', help='Force download')
    
    args = parser.parse_args()
    
    download_data_callback = DownloadData()
    download_data_callback.download_data(args.image_dir, args.bucket_dir, args.force)
