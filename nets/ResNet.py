from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import Distillation as Dist

def ResNet_arg_scope(weight_decay=0.0005):
    with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected], 
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_OUT'),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        biases_initializer=None, activation_fn = None,
                        ):
        with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                            scale = True, center = True, activation_fn=tf.nn.relu, decay=0.9, epsilon = 1e-5,
                            param_regularizers={'gamma': tf.contrib.layers.l2_regularizer(weight_decay),
                                                'beta' : tf.contrib.layers.l2_regularizer(weight_decay)
                                                },
                            variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'BN_collection']) as arg_sc:
            return arg_sc

def ResBlock(x, depth, stride, get_feat, is_training, reuse, name):
    with tf.variable_scope(name):
        out = tf.contrib.layers.conv2d(x,   depth, [3,3], stride, scope='conv0', trainable=is_training, reuse = reuse)
        out = tf.contrib.layers.batch_norm(out, scope='bn0', trainable = is_training, is_training=is_training, reuse = reuse)
        out = tf.contrib.layers.conv2d(out, depth, [3,3], 1,      scope='conv1', trainable=is_training, reuse = reuse)
        out = tf.contrib.layers.batch_norm(out, scope='bn1', trainable = is_training, is_training=is_training, reuse = reuse, activation_fn = None)
                
        if stride > 1 or depth != x.get_shape().as_list()[-1]:
            x = tf.contrib.layers.conv2d(x, depth, [1,1], stride, scope='conv2', trainable=is_training, reuse = reuse)
            x = tf.contrib.layers.batch_norm(x, scope='bn2', trainable = is_training, is_training=is_training, reuse = reuse, activation_fn = None)
        out = x+out
        if get_feat:
            tf.add_to_collection('feat_noact', out)
        out = tf.nn.relu(out)
        if get_feat:
            tf.add_to_collection('feat', out)
        return out
    
def NetworkBlock(x, nb_layers, depth, stride, is_training = False, reuse = False, name = ''):
    with tf.variable_scope(name):
        for i in range(nb_layers):           
            x = ResBlock(x, depth, stride = stride if i == 0 else 1,
                         get_feat = True if i == nb_layers-1 else False,
                         is_training = is_training, reuse = reuse, name = 'BasicBlock%d'%i)
        return x

def ResNet(image, scope, is_training=False, reuse = False, drop = False, Distill = None):
    end_points = {}
    
    nChannels = [32, 64, 128, 256]
    stride = [1,2,2]
    n = 1
    with tf.variable_scope(scope):
        std = tf.contrib.layers.conv2d(image, nChannels[0], [3,3], 1, scope='conv0', trainable=is_training, reuse = reuse)
        std = tf.contrib.layers.batch_norm(std, scope='bn0', trainable = is_training, is_training=is_training, reuse = reuse)
        for i in range(3):            
            std = NetworkBlock(std, n, nChannels[1+i], stride[i], is_training = is_training, reuse = reuse, name = 'Resblock%d'%i)
        fc = tf.reduce_mean(std, [1,2])
        logits = tf.contrib.layers.fully_connected(fc , 100,
                                      weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0),
                                      biases_initializer = tf.zeros_initializer(),
                                      biases_regularizer = tf.contrib.layers.l2_regularizer(5e-4),
                                      trainable=is_training, scope = 'full', reuse = reuse)
        end_points['Logits'] = logits
    
    if is_training and Distill is not None:
        with tf.variable_scope('Teacher'):
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected], weights_regularizer = None):
                with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],param_regularizers=None):
                    n = 5
                    tch = tf.contrib.layers.conv2d(image, nChannels[0], [3,3], 1, scope='conv0', trainable=False, reuse = reuse)
                    tch = tf.contrib.layers.batch_norm(tch, scope='bn0', trainable = False, is_training=False, reuse = reuse)
                    for i in range(3):            
                        tch = NetworkBlock(tch, n, nChannels[1+i], stride[i], is_training = False, reuse = reuse, name = 'Resblock%d'%i)
                    fc = tf.reduce_mean(tch, [1,2])
                    logits_tch = tf.contrib.layers.fully_connected(fc , 100, biases_initializer = tf.zeros_initializer(),
                                                      trainable=False, scope = 'full', reuse = False)
                    end_points['Logits_tch'] = logits_tch
        with tf.variable_scope('Distillation'):
            feats = tf.get_collection('feat')
            student_feats = feats[:len(feats)//2]
            teacher_feats = feats[len(feats)//2:]
            feats_noact = tf.get_collection('feat_noact')
            student_feats_noact = feats[:len(feats_noact)//2]
            teacher_feats_noact = feats[len(feats_noact)//2:]
            
            if Distill == 'Soft_logits':
                end_points['Dist'] = Dist.Soft_logits(logits, logits_tch, 3)
            elif Distill == 'FitNet':
                end_points['Dist'] = Dist.FitNet(student_feats, teacher_feats)
            elif Distill == 'AT':
                end_points['Dist'] = Dist.Attention_transfer(student_feats_noact, teacher_feats_noact)
            elif Distill == 'FSP':
                end_points['Dist'] = Dist.FSP(student_feats, teacher_feats)
            elif Distill == 'KD-SVD':
                end_points['Dist'] = Dist.KD_SVD(student_feats, teacher_feats)
            elif Distill == 'AB':
                end_points['Dist'] = Dist.AB_distillation(student_feats_noact, teacher_feats_noact, 1., 1e-3)
            elif Distill == 'RKD':
                end_points['Dist'] = Dist.RKD(logits, logits_tch)

            tf.add_to_collection('dist', end_points['Dist'])
    return end_points

