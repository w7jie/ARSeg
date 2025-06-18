import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.AttentionBlock import CrossAttention
from Config import Batch_Size

from utils.Conv import Conv_Block,Down_Sample,Up_Sample
from utils.Calculate import cal_dice_loss,cal_proto_loss,cal_kl_loss,cal_iou


class ARSeg_Net(nn.Module):
    def __init__(self,Batch_Size=8):
        super(ARSeg_Net, self).__init__()
        self.imgconv1 = Conv_Block(in_channels=3,out_channels=64,type='img')
        self.imgconv2 = Conv_Block(in_channels=64,out_channels=128,type='img')
        self.imgconv3 = Conv_Block(in_channels=128,out_channels=256,type='img')
        self.imgconv4 = Conv_Block(in_channels=256,out_channels=512,type='img')
        self.down = Down_Sample()
        self.textconv1 = Conv_Block(in_channels=768,out_channels=512,type='text')
        self.textconv2 = Conv_Block(in_channels=768, out_channels=512,type='text')
        self.textconv3 = Conv_Block(in_channels=768, out_channels=512,type='text')

        self.trans_text1=Conv_Block(in_channels=5,out_channels=196,type='text')
        self.trans_text2=Conv_Block(in_channels=5,out_channels=196,type='text')
        self.trans_text3=Conv_Block(in_channels=10,out_channels=196,type='text')

        self.imgtokens_Crossattn=CrossAttention(dim=512)
        self.text1tokens_Crossattn=CrossAttention(dim=512)
        self.text2tokens_Crossattn=CrossAttention(dim=512)
        self.text3tokens_Crossattn=CrossAttention(dim=512)
        self.img_Crossattn=CrossAttention(dim=512)
        self.text1_Crossattn=CrossAttention(dim=512)
        self.text2_Crossattn=CrossAttention(dim=512)
        self.text3_Crossattn=CrossAttention(dim=512)

        self.shared_up1=Up_Sample(512)
        self.shared_conv1=Conv_Block(in_channels=512,out_channels=256,type='img')
        self.shared_up2=Up_Sample(256)
        self.shared_conv2=Conv_Block(in_channels=256,out_channels=128,type='img')
        self.shared_up3=Up_Sample(128)
        self.shared_conv3=Conv_Block(in_channels=128,out_channels=64,type='img')
        self.shared_out=nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)

        self.th=nn.Sigmoid()

        self.img_tokens=nn.Parameter(torch.zeros(Batch_Size,14,512))
        self.text1_tokens=nn.Parameter(torch.zeros(Batch_Size,14,512))
        self.text2_tokens=nn.Parameter(torch.zeros(Batch_Size,14,512))
        self.text3_tokens=nn.Parameter(torch.zeros(Batch_Size,14,512))
        self.Wimg=nn.Parameter(torch.zeros(Batch_Size,14,512))
        self.Wtext1=nn.Parameter(torch.zeros(Batch_Size,14,512))
        self.Wtext2=nn.Parameter(torch.zeros(Batch_Size,14,512))
        self.Wtext3=nn.Parameter(torch.zeros(Batch_Size,14,512))

        self.initialize_weights()

    def initialize_weights(self):#高斯分布初始化权重
        torch.nn.init.normal_(self.img_tokens,std=0.02)
        torch.nn.init.normal_(self.text1_tokens,std=0.02)
        torch.nn.init.normal_(self.text2_tokens,std=0.02)
        torch.nn.init.normal_(self.text3_tokens,std=0.02)
        torch.nn.init.normal_(self.Wimg,std=0.02)
        torch.nn.init.normal_(self.Wtext1,std=0.02)
        torch.nn.init.normal_(self.Wtext2,std=0.02)
        torch.nn.init.normal_(self.Wtext3,std=0.02)

    def forward(self, image, text1, text2, text3, label , text_mask=None,beta=None,config=None):#image_size=B,3,224,224,text_size=B,N,768
        eps=1e-5
        if text_mask is None:
            text_mask = [1, 1, 1]
        image1=self.down(self.imgconv1(image))#B,64,112,112
        image2=self.down(self.imgconv2(image1))#,B,128,56,56
        image3=self.down(self.imgconv3(image2))#B,256,28,28
        image4=self.down(self.imgconv4(image3))#image_size=B,512,14,14

        text1=self.trans_text1(text1)#text_size=B,196,768
        text1_1=self.textconv1(text1.transpose(-1,-2)).transpose(-1,-2)#text_size=B,196,512

        text2=self.trans_text2(text2)
        text2_1=self.textconv2(text2.transpose(-1,-2)).transpose(-1,-2)

        text3=self.trans_text3(text3)
        text3_1=self.textconv3(text3.transpose(-1,-2)).transpose(-1,-2)

        token_update_time=7
        img=image4.flatten(-2,-1).transpose(-1,-2).clone()
        temp_imgtokens=self.img_Crossattn(self.img_tokens,img)
        for i in range(token_update_time):
            img = self.img_Crossattn(img, temp_imgtokens)  # 原模态信息做q,token做kv,并更新原模态信息
            temp_imgtokens=self.imgtokens_Crossattn(temp_imgtokens,img)#token做q,原模态信息做kv,并更新token

        #if text_mask[0] == 1:
        temp_text1tokens=self.text1tokens_Crossattn(self.text1_tokens,text1_1)
        text1_updated=text1_1.clone()
        for i in range(token_update_time):
            text1_updated = self.text1_Crossattn(text1_updated, temp_text1tokens)
            temp_text1tokens=self.text1tokens_Crossattn(temp_text1tokens,text1_updated)


        #if text_mask[1] == 1:
        temp_text2tokens=self.text2tokens_Crossattn(self.text2_tokens,text2_1)
        text2_updated=text2_1.clone()
        for i in range(token_update_time):
            text2_updated = self.text2_Crossattn(text2_updated, temp_text2tokens)
            temp_text2tokens=self.text2tokens_Crossattn(temp_text2tokens,text2_updated)

        #if text_mask[2] == 1:
        temp_text3tokens=self.text3tokens_Crossattn(self.text3_tokens,text3_1)
        text3_updated=text3_1.clone()
        for i in range(token_update_time):
            text3_updated = self.text3_Crossattn(text3_updated, temp_text3tokens)
            temp_text3tokens=self.text3tokens_Crossattn(temp_text3tokens,text3_updated)

        M_img=torch.matmul(temp_imgtokens.transpose(-1,-2).unsqueeze(3),self.Wimg.transpose(-1,-2).unsqueeze(2))

        #shared_decoder

        M_text1 = torch.matmul(temp_text1tokens.transpose(-1, -2).unsqueeze(3),self.Wtext1.transpose(-1, -2).unsqueeze(2))
        Tokens_mix1 = M_img + M_text1
        TM1_1=self.shared_conv1(self.shared_up1(Tokens_mix1,image3))#B,256,28,28
        TM1_2=self.shared_conv2(self.shared_up2(TM1_1,image2))#B,128,56,56
        TM1_3=self.shared_conv3(self.shared_up3(TM1_2,image1))#B,64,112,112
        TM1_out=self.shared_out(F.interpolate(TM1_3,scale_factor=2,mode='nearest'))
        TM1_4=self.th(TM1_out)#B,1,224,224


        M_text2 = torch.matmul(temp_text2tokens.transpose(-1, -2).unsqueeze(3),self.Wtext2.transpose(-1, -2).unsqueeze(2))
        Tokens_mix2 = M_img + M_text2
        TM2_1=self.shared_conv1(self.shared_up1(Tokens_mix2,image3))
        TM2_2=self.shared_conv2(self.shared_up2(TM2_1,image2))
        TM2_3=self.shared_conv3(self.shared_up3(TM2_2,image1))
        TM2_out=self.shared_out(F.interpolate(TM2_3,scale_factor=2,mode='nearest'))
        TM2_4=self.th(TM2_out)


        M_text3 = torch.matmul(temp_text3tokens.transpose(-1, -2).unsqueeze(3),self.Wtext3.transpose(-1, -2).unsqueeze(2))
        Tokens_mix3 = M_img + M_text3
        TM3_1=self.shared_conv1(self.shared_up1(Tokens_mix3,image3))
        TM3_2=self.shared_conv2(self.shared_up2(TM3_1,image2))
        TM3_3=self.shared_conv3(self.shared_up3(TM3_2,image1))
        TM3_out=self.shared_out(F.interpolate(TM3_3,scale_factor=2,mode='nearest'))
        TM3_4=self.th(TM3_out)

        Tokens_mixall = M_img+(text_mask[0]==1)*M_text1+(text_mask[1]==1)*M_text2+(text_mask[2]==1)*M_text3  # B,512,14,14

        TMa_1=self.shared_conv1(self.shared_up1(Tokens_mixall,image3))
        TMa_2=self.shared_conv2(self.shared_up2(TMa_1,image2))
        TMa_3=self.shared_conv3(self.shared_up3(TMa_2,image1))
        TMa_out=self.shared_out(F.interpolate(TMa_3,scale_factor=2,mode='nearest'))
        TMa_4=self.th(TMa_out)

        #calculate kl_loss
        kl_loss_tensor = torch.stack([
            torch.stack([
                cal_kl_loss(TM1_1, TMa_1, temp=2.0),
                cal_kl_loss(TM1_2, TMa_2, temp=2.0),
                cal_kl_loss(TM1_3, TMa_3, temp=2.0),
                cal_kl_loss(TM1_4, TMa_4, temp=2.0)
            ]),
            torch.stack([
                cal_kl_loss(TM2_1, TMa_1, temp=2.0),
                cal_kl_loss(TM2_2, TMa_2, temp=2.0),
                cal_kl_loss(TM2_3, TMa_3, temp=2.0),
                cal_kl_loss(TM2_4, TMa_4, temp=2.0)
            ]),
            torch.stack([
                cal_kl_loss(TM3_1, TMa_1, temp=2.0),
                cal_kl_loss(TM3_2, TMa_2, temp=2.0),
                cal_kl_loss(TM3_3, TMa_3, temp=2.0),
                cal_kl_loss(TM3_4, TMa_4, temp=2.0)
            ])
        ], dim=0).to(device='cuda')

        proto1_loss, dist1 = cal_proto_loss(TM1_4, TMa_4, label)
        proto2_loss, dist2 = cal_proto_loss(TM2_4, TMa_4, label)
        proto3_loss, dist3 = cal_proto_loss(TM3_4, TMa_4, label)

        eps=1e-5
        avg_dn = ((text_mask[0] == 1).float()*torch.sum(dist1) + (text_mask[0] == 1).float()*torch.sum(dist2) + (text_mask[0] == 1).float()*torch.sum(dist3)) / (torch.sum(text_mask) + eps)
        rp1 = torch.sum(dist1) / (avg_dn+eps)
        rp2 = torch.sum(dist2) / (avg_dn+eps)
        rp3 = torch.sum(dist3) / (avg_dn+eps)

        total_proto_loss = (text_mask[0] == 1 ).float()*rp1  * proto1_loss + (
                    text_mask[1] == 1).float()*rp2  * proto2_loss + (
                                       text_mask[2] == 1 ).float() *rp3 * proto3_loss

        temp_kl = kl_loss_tensor * beta[:, None] * text_mask[:, None]
        total_kl_loss = torch.sum(temp_kl)
        seg_loss = cal_dice_loss(TMa_4, label)
        loss_fun =config.lamda2*total_proto_loss+config.lamda1*total_kl_loss+seg_loss
        dice=1-seg_loss
        iou=cal_iou(TMa_4, label)

        return loss_fun ,TMa_4,dice,iou