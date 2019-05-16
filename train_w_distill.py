import tensorflow as tf

from tensorflow import ConfigProto
from tensorflow.keras.datasets.cifar100 import load_data

import time, os
import scipy.io as sio
import numpy as np
from random import shuffle

from nets import nets_factory
import op_util

home_path = os.path.dirname(os.path.abspath(__file__))

tf.app.flags.DEFINE_string('train_dir', 'test',
                           'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string('Distillation', 'RKD',
                           'Distillation method : Soft_logits, FitNet, AT, FSP, KD-SVD, AB, RKD')
tf.app.flags.DEFINE_string('teacher', 'ResNet32',
                           'pretrained teacher`s weight')
tf.app.flags.DEFINE_string('main_scope', 'Student',
                           'networ`s scope')
FLAGS = tf.app.flags.FLAGS
def main(_):
    ### define path and hyper-parameter
    model_name   = 'ResNet'
    Learning_rate =1e-1# initialization methods : 1e-2, others : 1e-1

    batch_size = 128
    val_batch_size = 200
    train_epoch = 0
    init_epoch = 40 if FLAGS.Distillation == 'FitNet' or FLAGS.Distillation == 'FSP' or FLAGS.Distillation == 'AB' else 0
    
    total_epoch = init_epoch + train_epoch
    weight_decay = 5e-4

    should_log          = 200
    save_summaries_secs = 20
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_num = '0'

    if FLAGS.Distillation == 'None':
        FLAGS.Distillation = None
    
    with tf.Graph().as_default() as graph:
        # make placeholder for inputs
        sz = [32,32,3]
        train_image = tf.placeholder(tf.uint8, [batch_size]+sz)
        train_label = tf.placeholder(tf.int32, [batch_size])
        
        # pre-processing
        image = pre_processing(train_image, is_training = True)
        label = tf.contrib.layers.one_hot_encoding(train_label, 100, on_value=1.0)
     
        # make global step
        global_step = tf.train.create_global_step()
        decay_steps = 50000 // batch_size
        epoch = tf.floor_div(tf.cast(global_step, tf.float32), decay_steps)
        max_number_of_steps = int(50000*total_epoch)//batch_size

        # make learning rate scheduler
        LR = learning_rate_scheduler(Learning_rate, [epoch, init_epoch, train_epoch], [0.3, 0.6, 0.8], 0.1)
        
        ## load Net
        class_loss, train_accuracy = MODEL(model_name, FLAGS.main_scope, weight_decay, image, label,
                                           is_training = True, reuse = False, drop = True, Distillation = FLAGS.Distillation)
        
        #make training operator
        train_op = op_util.Optimizer_w_Distillation(class_loss, LR, epoch, init_epoch, global_step, FLAGS.Distillation)
        
        ## collect summary ops for plotting in tensorboard
        summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES), name='summary_op')
        
        ## make clone model to validate
        val_image = tf.placeholder(tf.float32, [val_batch_size]+image.get_shape().as_list()[1:])
        val_label = tf.placeholder(tf.int32, [val_batch_size])
        val_label_onhot = tf.contrib.layers.one_hot_encoding(val_label, 100,on_value=1.0)
        val_image_ = pre_processing(val_image, is_training = False)
        val_loss, val_accuracy = MODEL(model_name, FLAGS.main_scope, 0., val_image_, val_label_onhot,
                                       is_training = False, reuse = True, drop = False, Distillation = FLAGS.Distillation)
        
        ## make placeholder and summary op for training and validation results
        train_acc_place = tf.placeholder(dtype=tf.float32)
        val_acc_place   = tf.placeholder(dtype=tf.float32)
        val_summary = [tf.summary.scalar('accuracy/training_accuracy',   train_acc_place),
                       tf.summary.scalar('accuracy/validation_accuracy', val_acc_place)]
        val_summary_op = tf.summary.merge(list(val_summary), name='val_summary_op')
        
        ## start training
        step = 0 ; highest = 0
        train_writer = tf.summary.FileWriter('%s'%FLAGS.train_dir,graph,flush_secs=save_summaries_secs)
        val_saver   = tf.train.Saver()
        config = ConfigProto()
        config.gpu_options.visible_device_list= gpu_num
        config.gpu_options.allow_growth=True
        
        (train_images, train_labels), (val_images, val_labels) = load_data()
        val_itr = len(val_labels)//val_batch_size
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
          
            if FLAGS.Distillation is not None:
                ## if Distillation is True, load and assign teacher's variables
                ## this mechanism is slower but easier to modifier than load checkpoint
                global_variables  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                p = sio.loadmat(home_path + '/pre_trained/%s.mat' %FLAGS.teacher)
                n = 0
                for v in global_variables:
                    if p.get(v.name[:-2]) is not None:
                        sess.run(v.assign(p[v.name[:-2]].reshape(*v.get_shape().as_list()) ))
                        n += 1
                print ('%d Teacher params assigned'%n)
                
            sum_train_accuracy = 0; time_elapsed = 0; total_loss = 0
            idx = list(range(train_labels.shape[0]))
            shuffle(idx)
            for step in range(max_number_of_steps):
                start_time = time.time()
                train_batch = train_images[idx[:batch_size]]
                
                ## tf.__version < 1.10
#                random_seed = [1]*(batch_size//2) + [-1]*(batch_size//2)
#                shuffle(random_seed)
#                train_batch = [ti if seed > 0 else np.fliplr(ti)
#                               for seed, ti in zip(random_seed, train_batch)]
                
                ## feed data
                tl, log, train_acc = sess.run([train_op, summary_op, train_accuracy],
                                              feed_dict = {train_image : train_batch,
                                                           train_label : np.squeeze(train_labels[idx[:batch_size]])}
                                              )
                time_elapsed += time.time() - start_time
                idx[:batch_size] = []
                if len(idx) < batch_size:
                    idx_ = list(range(train_labels.shape[0]))
                    shuffle(idx_)
                    idx += idx_

                total_loss += tl
                sum_train_accuracy += train_acc
                if ( step % (decay_steps) == 0) and (step//decay_steps>=init_epoch):
                    ## do validation
                    sum_val_accuracy = 0
                    
                    for i in range(val_itr):
                        val_batch = val_images[i*val_batch_size:(i+1)*val_batch_size]
                        acc = sess.run(val_accuracy, feed_dict = {val_image : val_batch,
                                                                  val_label : np.squeeze(val_labels[i*val_batch_size:(i+1)*val_batch_size]) })
                        sum_val_accuracy += acc

                    tf.logging.info('Epoch %s Step %s - train_Accuracy : %.2f%%  val_Accuracy : %.2f%%'
                                    %(str((step)//decay_steps).rjust(3, '0'), str(step).rjust(6, '0'), 
                                    sum_train_accuracy *100/decay_steps if step//decay_steps>init_epoch else 1., 
                                    sum_val_accuracy *100/val_itr))

                    result_log = sess.run(val_summary_op, feed_dict={
                                                                     train_acc_place : sum_train_accuracy*100/decay_steps if step//decay_steps>init_epoch else 1.,
                                                                     val_acc_place   : sum_val_accuracy*100/val_itr,
                                                                     })
                    if step//decay_steps == init_epoch and init_epoch > 0:
                        #re-initialize Momentum for fair comparison w/ initialization and multi-task learning methods
                        for v in global_variables:
                            if v.name[:-len('Momentum:0')]=='Momentum:0':
                                sess.run(v.assign(np.zeros(*v.get_shape().as_list()) ))
                                
                    if step == max_number_of_steps-1:
                        train_writer.add_summary(result_log, train_epoch)
                    else:
                        train_writer.add_summary(result_log, (step)//decay_steps-init_epoch)
                    sum_train_accuracy = 0
                    if sum_val_accuracy > highest:
                        highest = sum_val_accuracy
                        var = {}
                        variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)+tf.get_collection('BN_collection')
                        for v in variables:
                            var[v.name[:-2]] = sess.run(v)
                        sio.savemat(FLAGS.train_dir + '/best_params.mat',var)

                    val_saver.save(sess, "%s/best_model.ckpt"%FLAGS.train_dir)
                    
                if (step % should_log == 0)&(step > 0):
                    tf.logging.info('global step %s: loss = %.4f (%.3f sec/step)',str(step).rjust(6, '0'), total_loss/should_log, time_elapsed/should_log)
                    time_elapsed = 0
                    total_loss = 0
                
                elif (step % (decay_steps//2) == 0):
                    train_writer.add_summary(log, step)

            ## save variables to use for something
            var = {}
            variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)+tf.get_collection('BN_collection')
            for v in variables:
                var[v.name[:-2]] = sess.run(v)
            sio.savemat(FLAGS.train_dir + '/train_params.mat',var)
            
            ## close all
            tf.logging.info('Finished training! Saving model to disk.')
            train_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.STOP))
            train_writer.close()

def MODEL(model_name, scope, weight_decay, image, label, is_training, reuse, drop, Distillation):
    network_fn = nets_factory.get_network_fn(model_name, weight_decay = weight_decay)
    end_points = network_fn(image, scope, is_training=is_training, reuse=reuse, drop = drop, Distill=Distillation)

    loss = tf.losses.softmax_cross_entropy(label,end_points['Logits'])
    accuracy = tf.contrib.metrics.accuracy(tf.to_int32(tf.argmax(end_points['Logits'], 1)), tf.to_int32(tf.argmax(label, 1)))
    return loss, accuracy
    
def pre_processing(image, is_training):
    with tf.variable_scope('preprocessing'):
        image = tf.to_float(image)
        image = (image-np.array([112.4776,124.1058,129.3773],dtype=np.float32).reshape(1,1,3))/np.array([70.4587,65.4312,68.2094],dtype=np.float32).reshape(1,1,3)
        if is_training:
            image = tf.image.random_flip_left_right(image) # tf.__version__ > 1.10
            image = tf.pad(image, [[0,0],[4,4],[4,4],[0,0]], 'REFLECT')
            image = tf.random_crop(image,[image.get_shape().as_list()[0],32,32,3])
    return image

def learning_rate_scheduler(Learning_rate, epochs, decay_point, decay_rate):
    with tf.variable_scope('learning_rate_scheduler'):
        e, ie, te = epochs
        for i, dp in enumerate(decay_point):
            Learning_rate = tf.cond(tf.greater_equal(e, ie + int(te*dp)), lambda : Learning_rate*decay_rate, 
                                                                          lambda : Learning_rate)
        tf.summary.scalar('learning_rate', Learning_rate)
        return Learning_rate

if __name__ == '__main__':
    tf.app.run()

