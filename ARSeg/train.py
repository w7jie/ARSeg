import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim
import Config as config
import logging
from Utils import read_text,generate_text_mask
from nets.ARSeg_Net import ARSeg_Net
from torch.utils.data import DataLoader
from load_dataset import ITLoader
from train_one_epoch import train_one_epoch

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def train_model():
    train_text1,train_text2,train_text3=read_text(config.train_dataset+'Train_text.xlsx')
    val_text1,val_text2,val_text3=read_text(config.val_dataset+'Val_text.xlsx')
    train_dataset=ITLoader(config.train_dataset,train_text1,train_text2,train_text3)
    val_dataset=ITLoader(config.val_dataset,val_text1,val_text2,val_text3)

    train_loader=DataLoader(train_dataset,batch_size=config.Batch_Size,shuffle=True,drop_last=True)
    val_loader=DataLoader(val_dataset,batch_size=config.Batch_Size,shuffle=True,drop_last=True)

    model=ARSeg_Net(Batch_Size=config.Batch_Size)

    learning_rate=config.learning_rate
    model=model.cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=config.weight_decay)

    best_epoch=1
    eps=1e-5
    max_dice=torch.tensor(0.0).cuda()
    beta=[1/(2*(1-p+eps)) for p in config.probability]
    beta=torch.tensor(beta)
    beta=beta.cuda()
    logger.info('\n text_mask probability={:.1f} {:.1f} {:.1f}'.format(config.probability[0], config.probability[1],config.probability[2]))
    logger.info('\n lamda1={} lamda2={}'.format(config.lamda1,config.lamda2))
    logger.info(config.session_name)
    for epoch in range(config.epoch):
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epoch + 1))

        model.train()

        _,_=train_one_epoch(train_loader,model,optimizer,config.probability,beta,logger)
        if epoch+1>=35 or (epoch+1)%10==0:
            torch.save(model.state_dict(),config.model_path)

        torch.cuda.empty_cache()
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_dice,val_loss=train_one_epoch(val_loader,model,optimizer,config.probability,beta,logger)
            if val_dice>max_dice: max_dice=val_dice

        logger.info('\n Val Mean dice:{} Val mean loss:{}'.format(val_dice.item(),val_loss.item()))
        logger.info('best dice:{}', format(max_dice.item()))
        torch.cuda.empty_cache()




if __name__ == '__main__':
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)
    logger = logger_config(log_path=config.logger_path)
    train_model()