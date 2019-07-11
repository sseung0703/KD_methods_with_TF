import subprocess
#'''
for i in range(5):
    subprocess.call('python Graph_KD/train_w_distill.py '
                   +'--train_dir=/home/cvip/Documents/tf/KD/GIT/CIFAR100/ResNet/MHGD/mhkd%d '%i
                   +'--Distillation=MHGD',
                    shell=True)
    
    print ('Training Done')
