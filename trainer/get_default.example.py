import time

config = type('', (), {})()

config.bs = 4
config.in_h = 512
config.in_w = 512
config.epochs = 100
config.m = True
config.lr = 0.0002
config.job_dir = 'gs://my-bucket/my_project/jobs/{}'.format(config.project, str(time.time()))
config.train_csv = 'gs://my-bucket/data/my_train.csv'
config.test_csv = 'gs://my-bucket/data/my_test.csv'
config.image_dir = None
config.model_dir = './trained_models'

config.bucket_dir = 'gs://my-bucket/data/my_project'