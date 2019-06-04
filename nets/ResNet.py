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
        out = tf.contrib.layers.conv2d(x,   depth, [3,3], stride, scope='conv0', trainable=True, reuse = reuse)
        out = tf.contrib.layers.batch_norm(out, scope='bn0', trainable = True, is_training=is_training, reuse = reuse)
        out = tf.contrib.layers.conv2d(out, depth, [3,3], 1,      scope='conv1', trainable=True, reuse = reuse)
        out = tf.contrib.layers.batch_norm(out, scope='bn1', trainable = True, is_training=is_training, reuse = reuse, activation_fn = None)
                
        if stride > 1 or depth != x.get_shape().as_list()[-1]:
            x = tf.contrib.layers.conv2d(x, depth, [1,1], stride, scope='conv2', trainable=True, reuse = reuse)
            x = tf.contrib.layers.batch_norm(x, scope='bn2', trainable = True, is_training=is_training, reuse = reuse, activation_fn = None)
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

def ResNet(image, label, scope, is_training, reuse = False, drop = False, Distill = None):
    end_points = {}
    
    nChannels = [32, 64, 128, 256]
    stride = [1,2,2]
        
    n = 1 if scope != 'Teacher' else 5
    with tf.variable_scope(scope):
        std = tf.contrib.layers.conv2d(image, nChannels[0], [3,3], 1, scope='conv0', trainable=True, reuse = reuse)
        std = tf.contrib.layers.batch_norm(std, scope='bn0', trainable = True, is_training=is_training, reuse = reuse)
        for i in range(len(stride)):            
            std = NetworkBlock(std, n, nChannels[1+i], stride[i], is_training = is_training, reuse = reuse, name = 'Resblock%d'%i)
        fc = tf.reduce_mean(std, [1,2])
        logits = tf.contrib.layers.fully_connected(fc , label.get_shape().as_list()[-1],
                                      weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0),
                                      biases_initializer = tf.zeros_initializer(),
                                      biases_regularizer = tf.contrib.layers.l2_regularizer(5e-4),
                                      trainable=True, scope = 'full', reuse = reuse)
        end_points['Logits'] = logits
    
    if Distill is not None:
        if Distill == 'DML':
            teacher_train = True
            weight_decay = 5e-4
        else:
            is_training = False
            teacher_train = False
            weight_decay = 0.
        with tf.variable_scope('Teacher'):
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
                                                weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                                                variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'Teacher']):
                with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                                                    param_regularizers={'gamma': tf.contrib.layers.l2_regularizer(weight_decay),
                                                                        'beta' : tf.contrib.layers.l2_regularizer(weight_decay)},
                                                    variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'Teacher']):
                    n = 5
                    tch = tf.contrib.layers.conv2d(image, nChannels[0], [3,3], 1, scope='conv0', trainable=teacher_train, reuse = reuse)
                    tch = tf.contrib.layers.batch_norm(tch, scope='bn0', trainable = teacher_train, is_training=is_training, reuse = reuse)
                    for i in range(len(stride)):            
                        tch = NetworkBlock(tch, n, nChannels[1+i], stride[i], is_training = is_training, reuse = reuse, name = 'Resblock%d'%i)
                    fc = tf.reduce_mean(tch, [1,2])
                    logits_tch = tf.contrib.layers.fully_connected(fc , label.get_shape().as_list()[-1],
                                                                   weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0),
                                                                   biases_initializer = tf.zeros_initializer(),
                                                                   biases_regularizer = tf.contrib.layers.l2_regularizer(5e-4),
                                                                   trainable=teacher_train, scope = 'full', reuse = reuse)
                    end_points['Logits_tch'] = logits_tch
                    
        with tf.variable_scope('Distillation'):
            feats = tf.get_collection('feat')
            student_feats = feats[:len(feats)//2]
            teacher_feats = feats[len(feats)//2:]
            feats_noact = tf.get_collection('feat_noact')
            student_feats_noact = feats[:len(feats_noact)//2]
            teacher_feats_noact = feats[len(feats_noact)//2:]
            
            if Distill == 'Soft_logits':
                tf.add_to_collection('dist', Dist.Soft_logits(logits, logits_tch, 3))
            elif Distill == 'FitNet':
                tf.add_to_collection('dist', Dist.FitNet(student_feats, teacher_feats))
            elif Distill == 'AT':
                tf.add_to_collection('dist', Dist.Attention_transfer(student_feats_noact, teacher_feats_noact))
            elif Distill == 'FSP':
                tf.add_to_collection('dist', Dist.FSP(student_feats, teacher_feats))
            elif Distill == 'DML':
                tf.add_to_collection('dist', Dist.DML(logits, logits_tch))
            elif Distill == 'KD-SVD':
                tf.add_to_collection('dist', Dist.KD_SVD(student_feats, teacher_feats, 'SVD'))
            elif Distill == 'KD-EID':
                tf.add_to_collection('dist', Dist.KD_SVD(student_feats, teacher_feats, 'EID'))
            elif Distill == 'AB':
                tf.add_to_collection('dist', Dist.AB_distillation(student_feats_noact, teacher_feats_noact, 1., 3e-3))
                tf.add_to_collection('dist', Dist.Soft_logits(logits, logits_tch, 3))
            elif Distill == 'RKD':
                tf.add_to_collection('dist', Dist.RKD(logits, logits_tch))
                
    return end_points

