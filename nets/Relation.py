import tensorflow as tf
from nets import SVP

def RKD(source, target, l = [1e2,2e2]):
    '''
    Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho.  
    relational knowledge distillation.
    arXiv preprint arXiv:1904.05068, 2019.
    '''
    with tf.variable_scope('Relational_Knowledge_distillation'):
        def Huber_loss(x,y):
            with tf.variable_scope('Huber_loss'):
                return tf.reduce_mean(tf.where(tf.less_equal(tf.abs(x-y), 1.), 
                                               tf.square(x-y)/2, tf.abs(x-y)-1/2))
            
        def Distance_wise_potential(x):
            with tf.variable_scope('DwP'):
                x_square = tf.reduce_sum(tf.square(x),-1)
                prod = tf.matmul(x,x,transpose_b=True)
                distance = tf.sqrt(tf.maximum(tf.expand_dims(x_square,1)+tf.expand_dims(x_square,0) -2*prod, 1e-12))
                mu = tf.reduce_sum(distance)/tf.reduce_sum(tf.where(distance > 0., tf.ones_like(distance), tf.zeros_like(distance)))
                return distance/(mu+1e-8)
            
        def Angle_wise_potential(x):
            with tf.variable_scope('AwP'):
                e = tf.expand_dims(x,0)-tf.expand_dims(x,1)
                e_norm = tf.nn.l2_normalize(e,2)
            return tf.matmul(e_norm, e_norm,transpose_b=True)

        source = tf.nn.l2_normalize(source,1)
        target = tf.nn.l2_normalize(target,1)
        distance_loss = Huber_loss(Distance_wise_potential(source),Distance_wise_potential(target))
        angle_loss    = Huber_loss(   Angle_wise_potential(source),   Angle_wise_potential(target))
        
        return distance_loss*l[0]+angle_loss*l[1]    
    
def MHGD(student_feature_maps, teacher_feature_maps):
    '''
    Seunghyun Lee, Byung Cheol Song.
    Graph-based Knowledge Distillation by Multi-head Self-attention Network.
    British Machine Vision Conference (BMVC) 2019
    '''
    with tf.variable_scope('MHGD'):
        with tf.contrib.framework.arg_scope([tf.contrib.layers.fully_connected], trainable = True,
                                            weights_initializer=tf.initializers.random_normal(),
                                            weights_regularizer=None, variables_collections = [tf.GraphKeys.GLOBAL_VARIABLES,'MHA']):
            with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm], activation_fn=None, trainable = True,
                                                param_regularizers = None, variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'MHA']):
                GNN_losses = []
                num_head = 8
                V_Tb = V_Sb = None
                num_feat = len(student_feature_maps)
                for i, sfm, tfm in zip(range(num_feat), student_feature_maps, teacher_feature_maps):
                    with tf.variable_scope('Compress_feature_map%d'%i):
                        Sigma_T, U_T, V_T = SVP.SVD_eid(tfm, 1, name = 'TSVD%d'%i)
                        _,       U_S, V_S = SVP.SVD_eid(sfm, 1, name = 'SSVD%d'%i)
                        V_S, mask = SVP.Align_rsv(V_T, V_S)
                        D = V_T.get_shape().as_list()[1]
                    
                        V_T = tf.reshape(V_T,[-1,D])
                        V_S = tf.reshape(V_S,[-1,D])

                    with tf.variable_scope('MHA%d'%i):
                        if i > 0:
                            _,D_, = V_Sb.get_shape().as_list()
                            D2 = (D+D_)//2
                            G_T = Attention_head(V_T, V_Tb, D2, num_head, 'Attention', is_training = True)
                            V_T_ = Estimator(V_Tb, G_T, D, num_head, 'Estimator')
                            tf.add_to_collection('MHA_loss', tf.reduce_mean(1-tf.reduce_sum(V_T_*V_T, -1)) )
                            
                            G_T = Attention_head(V_T, V_Tb, D2, num_head, 'Attention', reuse = True)
                            G_S = Attention_head(V_S, V_Sb, D2, num_head, 'Attention', reuse = True)

                            mean = tf.reduce_mean(G_T, -1, keepdims=True)
                            G_T = tf.tanh(G_T-mean)
                            G_S = tf.tanh(G_S-mean)
                       
                            GNN_losses.append(kld_loss(G_S, G_T))
                            
                    V_Tb = V_T
                    V_Sb = V_S
        
                transfer_loss =  tf.add_n(GNN_losses)
        
                return transfer_loss
            
def Attention_head(K, Q, D, num_head, name, is_training = False, reuse = False):
    sz = tf.shape(K)
    B = tf.squeeze(tf.slice(sz,[0],[1]))
    
    with tf.variable_scope(name):
        X_sender   = tf.contrib.layers.fully_connected(K, D*num_head, scope = 'Sfc', reuse = reuse)
        X_sender   = tf.contrib.layers.batch_norm(X_sender, scope = 'Sbn', is_training = is_training, reuse = reuse)
        X_sender   = tf.reshape(X_sender,   [B, D, num_head])

        X_receiver = tf.contrib.layers.fully_connected(Q, D*num_head, scope = 'Rfc', reuse = reuse)
        X_receiver = tf.contrib.layers.batch_norm(X_receiver, scope = 'Rbn', is_training = is_training, reuse = reuse)
        X_receiver = tf.reshape(X_receiver, [B, D, num_head])
        
        X_sender   = tf.transpose(X_sender,  [2,0,1])
        X_receiver = tf.transpose(X_receiver,[2,1,0])
        X_ah = tf.matmul(X_sender, X_receiver)

    return X_ah

def Estimator(X, G, Dy, num_head, name):
    Dx = X.get_shape().as_list()[-1]
    B = tf.squeeze(tf.slice(tf.shape(G),[1],[1]))

    G = tf.nn.softmax(G)
    G = drop_head(G, [num_head, B, 1])
    G = tf.reshape(G, [num_head*B, B])

    D = (Dx+Dy)//2
    with tf.variable_scope(name):
        X = tf.contrib.layers.fully_connected(X, D, scope = 'fc0')
        X = tf.contrib.layers.batch_norm(X, activation_fn = tf.nn.relu, scope = 'bn0', is_training = True)

        X = tf.matmul(G, X)
        X = tf.reshape(tf.transpose(tf.reshape(X, [num_head, B, D]),[1,0,2]),[B,D*num_head])

        X = tf.contrib.layers.fully_connected(X, Dy, biases_initializer=tf.zeros_initializer(), scope = 'fc1')
        X = tf.nn.l2_normalize(X, -1)

    return X

def drop_head(G, shape):
    with tf.variable_scope('Drop'):
        noise = tf.random.normal(shape)
        G *= tf.where(noise - tf.reduce_mean(noise, 0, keepdims=True) > 0, tf.ones_like(noise), tf.zeros_like(noise))
        return G*2 - tf.stop_gradient(G)

def kld_loss(X, Y):
    with tf.variable_scope('KLD'):
        return tf.reduce_sum( tf.nn.softmax(X)*(tf.nn.log_softmax(X)-tf.nn.log_softmax(Y)) )
