import tensorflow as tf
import numpy as np
from nets import SVP

def Soft_logits(student, teacher, T = 2):
    '''
    Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.  
    Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.
    '''
    with tf.variable_scope('KD'):
        return tf.reduce_sum( tf.nn.softmax(teacher/T)*(tf.nn.log_softmax(teacher/T)-tf.nn.log_softmax(student/T)) )/student.get_shape().as_list()[0]

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
    B = student[0].get_shape().as_list()[0]
    return tf.add_n([Guided(std, tch) for i, std, tch in zip(range(len(student)), student, teacher)])
      
def FSP(students, teachers):
    '''
    Junho Yim, Donggyu Joo, Jihoon Bae, and Junmo Kim.
    A gift from knowledge distillation: Fast optimization, network minimization and transfer learning. 
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4133–4141, 2017.
    '''
    def Grammian(top, bot):
        with tf.variable_scope('Grammian'):
            t_sz = top.get_shape().as_list()
            b_sz = bot.get_shape().as_list()
    
            if t_sz[1] > b_sz[1]:
                top = tf.contrib.layers.max_pool2d(top, [2, 2], 2)
                            
            top = tf.reshape(top,[t_sz[0], -1, t_sz[-1]])
            bot = tf.reshape(bot,[b_sz[0], -1, b_sz[-1]])
    
            Gram = tf.matmul(top, bot, transpose_a = True)/(b_sz[1]*b_sz[2])
            return Gram, t_sz[-1]*b_sz[-1]
    with tf.variable_scope('FSP'):
        N = 0                    
        Dist_loss = []
        for i in range(len(students)-1):
            gs0, _ = Grammian(students[i], students[i+1])
            gt0, n = Grammian(teachers[i], teachers[i+1])
     
            Dist_loss.append(tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(gt0)-gs0),[1,2])/2 ))
            N += n

        return tf.add_n(Dist_loss)/N    

def KD_SVD(student_feature_maps, teacher_feature_maps):
    '''
    Seung Hyun Lee, Dae Ha Kim, and Byung Cheol Song.
    Self-supervised knowledge distillation using singular value decomposition. In
    European Conference on ComputerVision, pages 339–354. Springer, 2018.
    '''
    with tf.variable_scope('Distillation'):
        GNN_losses = []
        K = 4
        V_Tb = V_Sb = None
        for i, sfm, tfm in zip(range(len(student_feature_maps)), student_feature_maps, teacher_feature_maps):
            with tf.variable_scope('Compress_feature_map%d'%i):
                Sigma_T, U_T, V_T = SVP.SVD(tfm, K, name = 'TSVD%d'%i)
                Sigma_S, U_S, V_S = SVP.SVD(sfm, K, name = 'SSVD%d'%i)
                B, D,_ = V_S.get_shape().as_list()
                V_S, U_S, V_T = SVP.Align_rsv(V_S, V_T, U_S, Sigma_T, K)
                Sigma_T = tf.expand_dims(Sigma_T,1)
                V_T *= Sigma_T
                V_S *= Sigma_T
                
            if i > 0:
                with tf.variable_scope('RBF%d'%i):    
                    S_rbf = tf.exp(-tf.square(tf.expand_dims(V_S,2)-tf.expand_dims(V_Sb,1))/8)
                    T_rbf = tf.exp(-tf.square(tf.expand_dims(V_T,2)-tf.expand_dims(V_Tb,1))/8)

                    l2loss = (S_rbf-tf.stop_gradient(T_rbf))**2
                    l2loss = tf.where(tf.is_finite(l2loss), l2loss, tf.zeros_like(l2loss))
                    GNN_losses.append(tf.reduce_sum(l2loss))
            V_Tb = V_T
            V_Sb = V_S

        transfer_loss =  tf.add_n(GNN_losses)

        return transfer_loss
      
def AB_distillation(student, teacher, margin=1. , weight = 1e-3):
    '''
    Byeongho Heo,  Minsik Lee,  Sangdoo Yun,  and Jin Young Choi.   
    Knowledge transfer via distillation of activation boundaries formed by hidden neurons.
    arXiv preprint arXiv:1811.03233, 2018.
    '''
    def criterion_alternative_L2(source, target, margin):
        with tf.variable_scope('criterion_alternative_L2'):
            Ds = source.get_shape().as_list()[-1]
            Dt = target.get_shape().as_list()[-1]
            if Ds != Dt:
                with tf.variable_scope('Map'):
                    target = tf.contrib.layers.fully_connected(target, Ds, biases_initializer = None, trainable=True, scope = 'fc')
                    target = tf.contrib.layers.batch_norm(target,scope='bn',is_training=True, trainable = True, activation_fn = None)
            
            loss = tf.square(source + margin) * tf.cast(tf.logical_and(source > -margin, target <= 0.), tf.float32)\
                  +tf.square(source - margin) * tf.cast(tf.logical_and(source <= margin, target >  0.), tf.float32)
            return tf.reduce_sum(tf.abs(loss))
    B = student[0].get_shape().as_list()[0]
    return tf.add_n([criterion_alternative_L2(std, tch, margin)/B/2**(-i)
                    for i, std, tch in zip(range(len(student)), student, teacher)])*weight