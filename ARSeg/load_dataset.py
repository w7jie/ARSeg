import torch
from bert_embedding import BertEmbedding
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class ITLoader(Dataset):
    def __init__(self, dataset_path,row_text1,row_text2,row_text3,img_size=224):
        self.dataset_path = dataset_path
        self.row_text1 = row_text1
        self.row_text2 = row_text2
        self.row_text3 = row_text3
        self.bert_embedding = BertEmbedding()
        self.img_path=os.path.join(self.dataset_path,'img')
        self.label_path=os.path.join(self.dataset_path,'labelcol')
        self.img_list=os.listdir(self.img_path)
        self.label_list=os.listdir(self.label_path)
        self.img_size=img_size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_filename=self.img_list[idx]
        label_filename = img_filename[: -3] + "png"
        img=cv2.imread(os.path.join(self.img_path,img_filename))
        img=cv2.resize(img,(self.img_size,self.img_size),interpolation=cv2.INTER_AREA)
        label=cv2.imread(os.path.join(self.label_path,label_filename),0)
        label=cv2.resize(label,(self.img_size,self.img_size),interpolation=cv2.INTER_NEAREST)
        label[label<=0]=0
        label[label>0]=1
        img=img.transpose(2,0,1)
        label=np.expand_dims(label, axis=0)
        text1=self.row_text1[label_filename].split('\n')
        text2=self.row_text2[label_filename].split('\n')
        text3=self.row_text3[label_filename].split('\n')
        text1_token=self.bert_embedding(text1)
        text2_token=self.bert_embedding(text2)
        text3_token=self.bert_embedding(text3)
        text1=np.array(text1_token[0][1])#N,768
        if text1.shape[0]>5:
            text1 = text1[:5, :]
        text2=np.array(text2_token[0][1])
        if text2.shape[0]>5:
            text2 = text2[:5, :]
        text3=np.array(text3_token[0][1])
        if text3.shape[0]>10:
            text3 = text3[:10, :]
        return img,label,text1,text2,text3,img_filename