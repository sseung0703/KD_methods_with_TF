import tensorflow as tf

def Soft_logits(student, teacher, T = 2):
    '''
    Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.  
    Distilling the knowledge in a neural network.
    arXiv preprint arXiv:1503.02531, 2015.
    '''
    with tf.variable_scope('KD'):
        return tf.reduce_mean(tf.reduce_sum( tf.nn.softmax(teacher/T)*(tf.nn.log_softmax(teacher/T)-tf.nn.log_softmax(student/T)),1 ))

def DML(student, teacher):
    '''
    Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu. 
    Deep mutual learning.
    IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    '''
    with tf.variable_scope('KD'):
        return (tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(teacher)*(tf.nn.log_softmax(teacher)-tf.nn.log_softmax(student)),1)) +
                tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(student)*(tf.nn.log_softmax(student)-tf.nn.log_softmax(teacher)),1)))/2

def Factor_Transfer(sfm, tfm):
    '''
    Jangho Kim, SeoungUK Park, Nojun Kwak.
    Paraphrasing Complex Network: Network Compression via Factor Transfer.
    Advances in Neural Information Processing Systems (NeurIPS). 2018.
    '''
    def Factor_transfer(X, rate, scope, reuse = False):
        with tf.variable_scope(scope):
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.conv2d_transpose], weights_regularizer=None,
                                                variables_collections = [tf.GraphKeys.GLOBAL_VARIABLES, 'Para']):
                D = tfm.get_shape().as_list()[-1]
                conv = tf.contrib.layers.conv2d(X,    int(D*rate**1), [3,3], 1,          scope='conv0', reuse = reuse)
                conv = tf.contrib.layers.conv2d(conv, int(D*rate**2), [3,3], int(1/rate),scope='conv1', reuse = reuse)
                conv = tf.contrib.layers.conv2d(conv, int(D*rate**3), [3,3], 1,          scope='conv2', reuse = reuse)
                
                if reuse:
                    return tf.nn.l2_normalize(conv, -1)
                
                deconv = tf.contrib.layers.conv2d_transpose(conv,   int(D*rate**2), [3,3], 1,          scope='convt0', reuse = reuse)
                deconv = tf.contrib.layers.conv2d_transpose(deconv, int(D*rate**1), [3,3], int(1/rate),scope='convt1', reuse = reuse)
                deconv = tf.contrib.layers.conv2d_transpose(deconv, D, [3,3], 1,  scope='convt2', reuse = reuse)
                return deconv

    with tf.variable_scope('Factor_Transfer'):
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d], trainable = True,
                                            biases_initializer = tf.zeros_initializer(), activation_fn = tf.nn.leaky_relu):
            rate = 0.5
            tfm_ = Factor_transfer(tfm, rate, 'Factor_transfer')
            tf.add_to_collection('Para_loss', tf.reduce_mean(tf.reduce_mean(tf.abs(tfm-tfm_),[1,2,3])))
                    
            F_T = Factor_transfer(tfm, rate, 'Factor_transfer', True)

            with tf.variable_scope('Translator'):
                D = tfm.get_shape().as_list()[-1]
                conv = tf.contrib.layers.conv2d(sfm,  int(D*rate**1), [3,3], 1,          scope='conv0')
                conv = tf.contrib.layers.conv2d(conv, int(D*rate**2), [3,3], int(1/rate),scope='conv1')
                conv = tf.contrib.layers.conv2d(conv, int(D*rate**3), [3,3], 1,          scope='conv2')
                F_S = tf.nn.l2_normalize(conv, -1)
            return tf.reduce_mean(tf.reduce_mean(tf.abs(F_T-F_S),[1,2,3]))
