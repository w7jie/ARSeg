import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.Calculate import cal_dice_loss,cal_proto_loss
from Utils import predict_output,generate_text_mask
import Config as config

def train_one_epoch(data_loader,model,optimizer,probability,beta,logger):
    total_dice=torch.tensor(0.0).cuda()
    total_loss=torch.tensor(0.0).cuda()
    for i, (image,label,text1,text2,text3,name) in enumerate(data_loader):
        text_mask=generate_text_mask(probability).cuda()
        image,label,text1,text2,text3=image,label,text1,text2,text3
        image,label,text1,text2,text3=image.float().cuda(),label.float().cuda(),text1.cuda(),text2.cuda(),text3.cuda()
        loss_fun,output,dice,iou=model(image,text1,text2,text3, label ,text_mask,beta,config)

        if model.training:
            optimizer.zero_grad()
            loss_fun.backward()
            optimizer.step()

        total_dice=total_dice+dice.item()
        total_loss=total_loss+loss_fun.item()
        logger.info('[ {0}/{1} ]:'.format(i+1,len(data_loader)))
        logger.info(' dice:{} loss:{}'.format(dice.item(),loss_fun.item()))
    avg_dice=total_dice/len(data_loader)
    total_loss=total_loss/len(data_loader)
    return avg_dice,total_loss