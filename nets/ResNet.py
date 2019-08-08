from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import Response, Multiple, Shared, Relation

tcf = tf.contrib.framework
tcl = tf.contrib.layers

def ResBlock(x, depth, stride, get_feat, name):
    with tf.variable_scope(name):
        out = tcl.batch_norm(tcl.conv2d(x,   depth, [3,3], stride, scope='conv0'), scope='bn0')
        out = tcl.batch_norm(tcl.conv2d(out, depth, [3,3], 1,      scope='conv1'), scope='bn1',activation_fn = None)
                
        if stride > 1 or depth != x.get_shape().as_list()[-1]:
            x = tcl.batch_norm(tcl.conv2d(x, depth, [1,1], stride, scope='conv2'), scope='bn2', activation_fn = None)
            
        out_ = x+out
        out = tf.nn.relu(out_)
        
        if get_feat:
            tf.add_to_collection('feat_noact', out_)
            tf.add_to_collection('feat', out)
        return out
    
def NetworkBlock(x, nb_layers, depth, stride, name = ''):
    with tf.variable_scope(name):
        for i in range(nb_layers):           
            x = ResBlock(x, depth, stride = stride if i == 0 else 1,
                         get_feat = True if i == nb_layers-1 else False, name = 'BasicBlock%d'%i)
        return x

def ResNet(image, label, scope, is_training, Distill = None):
    end_points = {}
    is_training, auxiliary_is_training = is_training

    if image.get_shape().as_list()[1] == 32:
        nChannels = [32, 64, 128, 256]
        stride = [1,2,2]
    else:
        nChannels = [16, 32, 64, 128, 256, 512]
        stride = [1,2,2,2,2]
        
    n = 1 if scope != 'Teacher' else 5
    
    with tf.variable_scope(scope):
        with tcf.arg_scope([tcl.conv2d, tcl.fully_connected, tcl.batch_norm], trainable = True):
            with tcf.arg_scope([tcl.dropout, tcl.batch_norm], is_training = is_training):
                std = tcl.conv2d(image, nChannels[0], [3,3], 1, scope='conv0')
                std = tcl.batch_norm(std, scope='bn0')
                for i in range(len(stride)):            
                    std = NetworkBlock(std, n, nChannels[1+i], stride[i], name = 'Resblock%d'%i)
                fc = tf.reduce_mean(std, [1,2])
                logits = tcl.fully_connected(fc , label.get_shape().as_list()[-1],
                                             biases_initializer = tf.zeros_initializer(),
                                             biases_regularizer = tcl.l2_regularizer(5e-4),
                                             scope = 'full')
                end_points['Logits'] = logits
        
    if Distill is not None:
        if Distill == 'DML':
            teacher_trainable = True
            weight_decay = 5e-4
            teacher_is_training = tf.logical_not(is_training)
        else:
            teacher_trainable = False
            weight_decay = 0.
            teacher_is_training = False
            
        arg_scope = ResNet_arg_scope_teacher(weight_decay=weight_decay)
            
        with tf.variable_scope('Teacher'):
            with tcf.arg_scope(arg_scope):
                with tcf.arg_scope([tcl.conv2d, tcl.fully_connected, tcl.batch_norm], trainable = teacher_trainable):
                    with tcf.arg_scope([tcl.batch_norm], is_training = teacher_is_training):
                        n = 5
                        tch = tcl.conv2d(image, nChannels[0], [3,3], 1, scope='conv0')
                        tch = tcl.batch_norm(tch, scope='bn0')
                        for i in range(len(stride)):            
                            tch = NetworkBlock(tch, n, nChannels[1+i], stride[i], name = 'Resblock%d'%i)
                        fc = tf.reduce_mean(tch, [1,2])
                        logits_tch = tcl.fully_connected(fc , label.get_shape().as_list()[-1],
                                                         biases_initializer = tf.zeros_initializer(),
                                                         biases_regularizer = tcl.l2_regularizer(weight_decay) if weight_decay > 0. else None,
                                                         scope = 'full')
                        end_points['Logits_tch'] = logits_tch
                    
        with tf.variable_scope('Distillation'):
            feats = tf.get_collection('feat')
            student_feats = feats[:len(feats)//2]
            teacher_feats = feats[len(feats)//2:]
            
            if Distill == 'Soft_logits':
                tf.add_to_collection('dist', Response.Soft_logits(logits, logits_tch, 4))
            elif Distill == 'DML':
                tf.add_to_collection('dist', Response.DML(logits, logits_tch))
            elif Distill == 'FT':
                tf.add_to_collection('dist', Response.Factor_Transfer(student_feats[-1], teacher_feats[-1]))
                
            elif Distill == 'FitNet':
                tf.add_to_collection('dist', Multiple.FitNet(student_feats, teacher_feats))
            elif Distill == 'AT':
                tf.add_to_collection('dist', Multiple.Attention_transfer(student_feats, teacher_feats))
            elif Distill == 'AB':
                tf.add_to_collection('dist', Multiple.AB_distillation(student_feats, teacher_feats, 1., 3e-3))
                
            elif Distill == 'FSP':
                tf.add_to_collection('dist', Shared.FSP(student_feats, teacher_feats))
            elif Distill[:3] == 'KD-':
                tf.add_to_collection('dist', Shared.KD_SVD(student_feats, teacher_feats, Distill[-3:]))

            elif Distill == 'RKD':
                tf.add_to_collection('dist', Relation.RKD(logits, logits_tch, l = [5e1,1e2]))
            elif Distill == 'MHGD':
                tf.add_to_collection('dist', Relation.MHGD(student_feats, teacher_feats))
                
    return end_points

def ResNet_arg_scope(weight_decay=0.0005):
    with tcf.arg_scope([tcl.conv2d, tcl.fully_connected], 
                       weights_initializer=tcl.variance_scaling_initializer(mode='FAN_OUT'),
                       weights_regularizer=tcl.l2_regularizer(weight_decay),
                       biases_initializer=None, activation_fn = None):
        with tcf.arg_scope([tcl.batch_norm], scale = True, center = True, activation_fn=tf.nn.relu, decay=0.9, epsilon = 1e-5,
                           param_regularizers={'gamma': tcl.l2_regularizer(weight_decay),
                                               'beta' : tcl.l2_regularizer(weight_decay)},
                           variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'BN_collection']) as arg_sc:
            return arg_sc
def ResNet_arg_scope_teacher(weight_decay=0.0005):
    with tcf.arg_scope([tcl.conv2d, tcl.fully_connected], 
                       weights_regularizer = tcl.l2_regularizer(weight_decay) if weight_decay > 0. else None,
                       variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'Teacher']):
        with tcf.arg_scope([tcl.batch_norm], 
                           param_regularizers={'gamma': tcl.l2_regularizer(weight_decay),
                                               'beta' : tcl.l2_regularizer(weight_decay)} if weight_decay > 0. else None,
                           variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'Teacher']) as arg_sc:
            return arg_sc
