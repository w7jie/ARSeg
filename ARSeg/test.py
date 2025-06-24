import torch
import os
import Config as config
from nets.ARSeg_Net import ARSeg_Net
from Utils import read_text,generate_text_mask,predict_output
from torch.utils.data import DataLoader
from utils.Calculate import cal_dice_loss
from load_dataset import ITLoader
import numpy as np
import cv2

visualization=False

if __name__ == '__main__':


    test_session=config.test_session
    test_text1,test_text2,test_session3=read_text(config.test_dataset+'Test_text.xlsx')
    test_dataset=ITLoader(config.test_dataset,test_text1,test_text2,test_session3)
    test_loader=DataLoader(dataset=test_dataset,batch_size=config.Batch_Size,shuffle=True,drop_last=True)

    model_path='./'+config.task_name+'/MTN/'+config.test_session+'/model.pth'
    vis_path='./'+config.task_name+'/visualization/'
    if not os.path.isdir(vis_path):
        os.makedirs(vis_path)
    model=ARSeg_Net(Batch_Size=config.Batch_Size)
    model=model.cuda()
    model.load_state_dict(torch.load(model_path))
    print('model loaded')
    dice,iou=[],[]
    eps=1e-5
    beta = [1 / (2 * (1 - p + eps)) for p in config.probability]
    beta=torch.tensor(beta)
    beta=beta.cuda()
    for i, (image, label, text1, text2, text3, name) in enumerate(test_loader):
        text_mask = generate_text_mask(config.probability).cuda()
        image, label, text1, text2, text3 = image, label, text1, text2, text3
        image, label, text1, text2, text3 = image.float().cuda(), label.float().cuda(), text1.cuda(), text2.cuda(), text3.cuda()
        with torch.no_grad():
            model.eval()
            _,out,temp_dice,temp_iou=model(image,text1,text2,text3, label ,text_mask,beta,config)
            if visualization==True:
                for i in range(out.shape[0]):
                    img=out[i].squeeze(0).cpu().detach().numpy()
                    img=((img>0.5)*255).astype(np.uint8)
                    cv2.imwrite(vis_path+name[i],img)
            dice.append(temp_dice.cpu().detach())
            iou.append(temp_iou.cpu().detach())
            print('temp_dice',temp_dice,'temp_iou',temp_iou)

    cal_dice=np.array(dice)
    cal_iou=np.array(iou)
    mdice=np.mean(cal_dice)
    miou=np.mean(iou)
    print('mdice:',mdice,'miou',miou)