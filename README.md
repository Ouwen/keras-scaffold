# Keras Scaffold

The following repo is a starting scaffold for modeling projects.

### Implemented Models
 - CycleGAN
 - Pix2Pix

### Implemented Blocks
 - Unet Attention Block
 - Residual Block

### Regularizations
 - TODO: focal loss
 - TODO: diversity sensitive GAN

### Implemented Callbacks
 - `log_code.py`, logs code to GCP bucket before training
 - `copy_keras_model.py`, copies keras models to GCP bucket
 - `save_multi_model.py`, saves multiple keras models each epoch

### Custom Callbacks
 - `generate_images.py`, writes images to tensorboard
 - `get_csv_metrics.py`, writes metrics of interest to a csv file.
 
### Hyperparam tuning
 - GCP hyperparam tuning
 - TODO: Axios
