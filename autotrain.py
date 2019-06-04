import subprocess

for i in range(5):
    subprocess.call('python Knowledge_Distillation/train_w_distill.py '
                   +'--train_dir=Test/test%d '%i
                   +'--Distillation=KD-EID',
                    shell=True)
    
    print ('Training Done')

