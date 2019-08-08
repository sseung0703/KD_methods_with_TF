import tensorflow as tf
import scipy.io as sio    
import numpy as np

def Dataloader(name, home_path, model_name):
    if name == 'cifar100':
        return Cifar100(home_path, model_name)

def Cifar100(home_path, model_name):
    from tensorflow.keras.datasets.cifar100 import load_data
    (train_images, train_labels), (val_images, val_labels) = load_data()
    teacher = sio.loadmat(home_path + '/pre_trained/%s.mat'%model_name)
    def pre_processing(image, is_training):
        with tf.variable_scope('preprocessing'):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([112.4776,124.1058,129.3773]))/np.array([70.4587,65.4312,68.2094])
            def augmentation(image):
                image = tf.image.random_flip_left_right(image) # tf.__version__ > 1.10
                sz = tf.shape(image)
                image = tf.pad(image, [[0,0],[4,4],[4,4],[0,0]], 'REFLECT')
                image = tf.random_crop(image,sz)
                return image
            image = tf.cond(is_training, lambda : augmentation(image), lambda : image)
        return image
    
    return train_images, train_labels, val_images, val_labels, pre_processing, teacher
