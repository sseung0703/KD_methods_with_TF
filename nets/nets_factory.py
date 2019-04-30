import functools

import tensorflow as tf

from nets import ResNet

networks_map   = {
                 'ResNet':ResNet.ResNet,
                 }

arg_scopes_map = {
                  'ResNet':ResNet.ResNet_arg_scope,
                 }

def get_network_fn(name, weight_decay=5e-4):
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    func = networks_map[name]
    @functools.wraps(func)
    def network_fn(images, is_training, reuse, drop, Distill):
        with tf.contrib.framework.arg_scope(arg_scope):
            return func(images, is_training=is_training, reuse = reuse, drop = drop, Distill=Distill)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn

