import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def Optimizer_w_Distillation(class_loss, LR, epoch, init_epoch, global_step):
    with tf.variable_scope('Optimizer_w_Distillation'):
        # get variables and update operations
        teacher_variables  = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name[:len('Teacher')] == 'Teacher']
        teacher_update_ops = [u for u in tf.get_collection(tf.GraphKeys.UPDATE_OPS)          if u.name[:len('Teacher')] == 'Teacher']
        teacher_reg_loss   = tf.add_n([l for l in tf.losses.get_regularization_losses()      if l.name[:len('Teacher')] == 'Teacher'])
        
        student_variables  = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name[:len('Teacher')] != 'Teacher']
        student_update_ops = [u for u in tf.get_collection(tf.GraphKeys.UPDATE_OPS)          if u.name[:len('Teacher')] != 'Teacher']
        student_reg_loss   = tf.add_n([l for l in tf.losses.get_regularization_losses()      if l.name[:len('Teacher')] != 'Teacher'])
        
        optimize = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True)
        teacher_loss = tf.get_collection('teacher_class_loss')[0] + teacher_reg_loss + tf.get_collection('dist')[0]
        student_loss = class_loss + student_reg_loss + tf.get_collection('dist')[0]
        
        tf.summary.scalar('loss/total_loss', student_loss)
        gradients_teacher = optimize.compute_gradients(teacher_loss, var_list = teacher_variables)
        gradients_student = optimize.compute_gradients(student_loss, var_list = student_variables)
        
        # merge update operators and make train operator
        teacher_update_ops.append(optimize.apply_gradients(gradients_teacher))
        teacher_update_op = tf.group(*teacher_update_ops)
        teacher_train_op = control_flow_ops.with_dependencies([teacher_update_op], teacher_loss, name='teacher_train_op')
        
        student_update_ops.append(optimize.apply_gradients(gradients_student, global_step=global_step))
        student_update_op = tf.group(*student_update_ops)
        student_train_op = control_flow_ops.with_dependencies([student_update_op], student_loss, name='student_train_op')
        
        return teacher_train_op, student_train_op
