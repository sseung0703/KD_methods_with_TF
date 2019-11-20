import subprocess
import tensorflow as tf
import glob
import scipy.io as sio
import numpy as np

base_path = 'Test/mhgd'
for i in range(5):
    subprocess.call('python KD_methods_with_TF/train_w_distill.py '
                   +'--train_dir=%s%d '%(base_path,i)
                   +'--model_name=ResNet '
                   +'--Distillation=MHGD',
                    shell=True)
    print ('Training Done')

pathes = glob.glob(base_path[:-len(base_path.split('/')[-1])] + '*')
training_acc   = []
validation_acc = []
for path in pathes:
    logs = sio.loadmat(path + '/log.mat')
    training_acc.append(logs['training_acc'])
    validation_acc.append(logs['validation_acc'])
training_acc   = np.mean(np.vstack(training_acc),0)
validation_acc = np.mean(np.vstack(validation_acc),0)

train_acc_place = tf.placeholder(dtype=tf.float32)
val_acc_place   = tf.placeholder(dtype=tf.float32)
val_summary = [tf.summary.scalar('accuracy/training_accuracy',   train_acc_place),
               tf.summary.scalar('accuracy/validation_accuracy', val_acc_place)]
val_summary_op = tf.summary.merge(list(val_summary), name='val_summary_op')
    
train_writer = tf.summary.FileWriter(base_path[:-len(base_path.split('/')[-1])] + 'average',flush_secs=1)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    for i, (train_acc, val_acc) in enumerate(zip(training_acc,validation_acc)):
        result_log = sess.run(val_summary_op, feed_dict={train_acc_place : train_acc,
                                                         val_acc_place   : val_acc   })
        train_writer.add_summary(result_log, i)
    train_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.STOP))
    train_writer.close()
