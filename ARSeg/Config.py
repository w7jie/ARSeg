import os
import torch
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
use_cuda = torch.cuda.is_available()


Batch_Size=8
learning_rate = 2e-4
weight_decay=1e-4

probability=[0.1,0.4,0.7]

lamda1=0.5
lamda2=0.5
gamma=0.01

epoch=100#MosMedDataPlus
#epoch=50#QaTa-Covid19

#task_name = 'QaTa-Covid19'
task_name = 'MosMedDataPlus'
model_name = 'ARSeg_Net'

train_dataset = './datasets/' + task_name + '/Train_Folder/'
val_dataset = './datasets/' + task_name + '/Val_Folder/'
test_dataset = './datasets/' + task_name + '/Test_Folder/'
task_dataset = './datasets/' + task_name + '/Train_Folder/'
session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + '/' + model_name+ '/' + session_name + '/'
model_path = save_path + 'model.pth'
logger_path = save_path + session_name + ".log"
test_session='Test_session_01.21_08h56'