import tensorflow as tf
import argparse
import trainer
from trainer import utils
from trainer import models
from trainer import callbacks


def main(args):
    # Load Data
    Dataset = trainer.Dataset(image_dir=args.image_dir)
    train_dataset, train_count = Dataset.get_dataset(
        csv=args.train_csv, 
        batch_size=args.bs, 
        shape=(args.in_h, args.in_w, 1))

    test_dataset, val_count = Dataset.get_dataset(
        csv=args.test_csv, 
        batch_size=args.bs,
        shape=(args.in_h, args.in_w, 1))
    
    # Select and Compile Model
    filters = [8, 16, 32, 64, 128, 256, 512]
    dfilters = [8, 16, 32, 64, 128, 256, 512]
    
    generator_model = models.LeakyUnetModel(shape=(None, None, 1),
                      Activation=tf.keras.layers.LeakyReLU(0.1),
                      filters=filters,
                      filter_shape=(3,3))()
    
    discriminator_model = models.PatchDiscriminatorModel(shape=(args.in_h, args.in_w, 1),
                                                         Activation=tf.keras.layers.LeakyReLU(0.1),
                                                         filters=dfilters,
                                                         filter_shape=(3,3))()
    
    model = models.Pix2Pix(verbose = 1,
                           shape = (None, None, 1),
                           g=generator_model,
                           d=discriminator_model)

    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr, 0.5),
                  d_loss=utils.mse,
                  g_loss = [utils.mse, utils.mae],
                  loss_weights = [1, 100],
                  metrics=[utils.mse, utils.mae, utils.ssim, utils.psnr])
    
    # Generate Callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=args.job_dir, write_graph=True, update_freq='epoch')
    prog_bar = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)
    saving = tf.keras.callbacks.ModelCheckpoint(args.model_dir + '/model.{epoch:02d}-{val_ssim:.10f}.hdf5', 
                                                monitor='val_psnr', verbose=1, period=1, mode='max')
    
    save_multi_model = callbacks.SaveMultiModel([('g', generator_model), ('d', discriminator_model)], args.model_dir)
    log_code = callbacks.LogCode(args.job_dir, './trainer')
    copy_keras = callbacks.CopyKerasModel(args.model_dir, args.job_dir)
    
    # Fit the model
    model.fit(train_dataset,
              steps_per_epoch=int(train_count/args.bs),
              epochs=args.epochs,
              validation_data=test_dataset,
              validation_steps=int(val_count/args.bs),
              callbacks=[log_code, tensorboard, saving, save_multi_model, copy_keras, prog_bar])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input parser
    parser.add_argument('--bs',       type=int,   help='batch size')
    parser.add_argument('--in_h',     type=int,   help='image input size height')
    parser.add_argument('--in_w',     type=int,   help='image input size width')
    parser.add_argument('--epochs',   type=int,   help='number of epochs')
    parser.add_argument('--m',        type=bool,  help='manual run or hp tuning')
    parser.add_argument('--lr',       type=float, help='learning rate')

    # Cloud ML Params
    parser.add_argument('--job-dir',              help='Job directory for Google Cloud ML')
    parser.add_argument('--model_dir',            help='Local model directory')
    parser.add_argument('--train_csv',            help='Train csv')
    parser.add_argument('--test_csv',             help='Test csv')
    parser.add_argument('--image_dir',            help='Local image directory')
    args = parser.parse_args()
    
    for key in vars(args):
        if parser.get_default(key) is not None:
            setattr(trainer.config, key, parser.get_default(key))
            
    main(trainer.config)
