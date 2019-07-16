import tensorflow as tf

def FitNet(student, teacher):
    '''
     Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta,  and  Yoshua  Bengio.
     Fitnets:   Hints  for  thin  deep  nets.
     arXiv preprint arXiv:1412.6550, 2014.
    '''
    def Guided(source, target):
        with tf.variable_scope('Guided'):
            Ds = source.get_shape().as_list()[-1]
            Dt = target.get_shape().as_list()[-1]
            if Ds != Dt:
                with tf.variable_scope('Map'):
                    target = tf.contrib.layers.fully_connected(target, Ds, biases_initializer = None, trainable=True, scope = 'fc')
            
            return tf.reduce_mean(tf.square(source-target))
    return tf.add_n([Guided(std, tch) for i, std, tch in zip(range(len(student)), student, teacher)])

def Attention_transfer(student, teacher, beta = 1e3):
    '''
     Zagoruyko, Sergey and Komodakis, Nikos.
     Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer.
     arXiv preprint arXiv:1612.03928, 2016.
    '''
    def Attention(source, target):
        with tf.variable_scope('Attention'):
            B,_,_,Ds = source.get_shape().as_list()
            Dt = target.get_shape().as_list()[-1]
            if Ds != Dt:
                with tf.variable_scope('Map'):
                    source = tf.contrib.layers.fully_connected(source, Ds, biases_initializer = None, trainable=True, scope = 'fc')
            
            Qt = tf.contrib.layers.flatten(tf.reduce_mean(tf.square(source),-1))
            Qt = tf.nn.l2_normalize(Qt, [1,2])
            
            Qs = tf.contrib.layers.flatten(tf.reduce_mean(tf.square(target),-1))
            Qs = tf.nn.l2_normalize(Qs, [1,2])
            
            return tf.reduce_mean(tf.square(Qt-Qs))*beta/2
    return tf.add_n([Attention(std, tch) for i, std, tch in zip(range(len(student)), student, teacher)])
    
def AB_distillation(student, teacher, margin=1., weight = 3e-3):
    '''
    Byeongho Heo,  Minsik Lee,  Sangdoo Yun,  and Jin Young Choi.   
    Knowledge transfer via distillation of activation boundaries formed by hidden neurons.
    AAAI Conference on Artificial Intelligence (AAAI), 2019.
    '''
    def criterion_alternative_L2(source, target, margin, num):
        with tf.variable_scope('criterion_alternative_L2'):
            Dt = target.get_shape().as_list()[-1]
            with tf.variable_scope('Map'):
                source = tf.contrib.layers.conv2d(source, Dt, [1, 1], biases_initializer = None, trainable=True, scope = 'connector%d' % (num))
                source = tf.contrib.layers.batch_norm(source, scope='connector_bn%d' %num, is_training=True, trainable = True, activation_fn = None)
            
            loss = tf.square(source + margin) * tf.cast(tf.logical_and(source > -margin, target <= 0.), tf.float32)\
                  +tf.square(source - margin) * tf.cast(tf.logical_and(source <= margin, target > 0.), tf.float32)
            return tf.reduce_mean(tf.reduce_sum(tf.abs(loss),[1,2,3]))
    return tf.add_n([criterion_alternative_L2(std, tch, margin, i)/2**(-i)
                    for i, std, tch in zip(range(len(student)), student, teacher)])*weight
    
