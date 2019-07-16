import tensorflow as tf
from nets import SVP

def FSP(students, teachers, weight = 1e-3):
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
                            
            top = tf.reshape(top,[-1, b_sz[1]*b_sz[2], t_sz[-1]])
            bot = tf.reshape(bot,[-1, b_sz[1]*b_sz[2], b_sz[-1]])
    
            Gram = tf.matmul(top, bot, transpose_a = True)/(b_sz[1]*b_sz[2])
            return Gram
    with tf.variable_scope('FSP'):
        Dist_loss = []
        for i in range(len(students)-1):
            gs0 = Grammian(students[i], students[i+1])
            gt0 = Grammian(teachers[i], teachers[i+1])
     
            Dist_loss.append(tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(gt0)-gs0),[1,2])/2 ))

        return tf.add_n(Dist_loss)*weight
    
def KD_SVD(student_feature_maps, teacher_feature_maps, dist_type = 'SVD'):
    '''
    Seung Hyun Lee, Dae Ha Kim, and Byung Cheol Song.
    Self-supervised knowledge distillation using singular value decomposition. In
    European Conference on ComputerVision, pages 339–354. Springer, 2018.
    '''
    with tf.variable_scope('Distillation'):
        GNN_losses = []
        K = 1
        V_Tb = V_Sb = None
        for i, sfm, tfm in zip(range(len(student_feature_maps)), student_feature_maps, teacher_feature_maps):
            with tf.variable_scope('Compress_feature_map%d'%i):
                if dist_type == 'SVD':
                    Sigma_T, U_T, V_T = SVP.SVD(tfm, K, name = 'TSVD%d'%i)
                    Sigma_S, U_S, V_S = SVP.SVD(sfm, K, name = 'SSVD%d'%i)
                    B, D,_ = V_S.get_shape().as_list()
                    V_S, mask = SVP.Align_rsv(V_T, V_S)
                    
                elif dist_type == 'EID':
                    Sigma_T, U_T, V_T = SVP.SVD_eid(tfm, K, name = 'TSVD%d'%i)
                    Sigma_S, U_S, V_S = SVP.SVD_eid(sfm, K, name = 'SSVD%d'%i)
                    B, D,_ = V_S.get_shape().as_list()
                    V_S, mask = SVP.Align_rsv(V_T, V_S)
                
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
