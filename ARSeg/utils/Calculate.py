import torch
import torch.nn as nn
import torch.nn.functional as F

def cal_kl_loss(input_s,input_t,temp=1.0):
    pred_s=F.log_softmax(input_s/temp,dim=1)
    pred_t=F.softmax(input_t/temp,dim=1)
    pred_s = torch.clamp(pred_s, min=0.005, max=1)
    pred_t = torch.clamp(pred_t, min=0.005, max=1)
    pred_s = torch.log(pred_s)
    kl_loss = temp * temp * torch.mul(pred_t, torch.log(pred_t) - pred_s)
    kl_loss = torch.mean(kl_loss)
    return kl_loss

def cal_dice_loss(output,ground,eps=1e-5):
    ground=ground.float()
    interaction=2*torch.sum((output*ground))
    den=torch.sum(output)+torch.sum(ground)+eps
    return 1.0-interaction/den

def cal_iou(output,ground,eps=1e-5):
    ground = ground.float()
    interaction = torch.sum((output * ground))
    den = torch.sum(output) + torch.sum(ground) + eps-interaction
    return interaction/den

def cal_proto_loss(feature_s,feature_t,label):
    label=label.float()
    eps=1e-5
    proto_s=torch.sum(feature_s*label,dim=(-2,-1),keepdim=True)/(torch.sum(label,dim=(-2,-1),keepdim=True)+eps)
    proto_t=torch.sum(feature_t*label,dim=(-2,-1),keepdim=True)/(torch.sum(label,dim=(-2,-1),keepdim=True)+eps)
    proto_map_s=F.cosine_similarity(feature_s,proto_s,dim=1,eps=eps)
    proto_map_t = F.cosine_similarity(feature_t, proto_t, dim=1, eps=eps)
    proto_loss = torch.mean((proto_map_s - proto_map_t) ** 2)
    dist = torch.mean(torch.sqrt((proto_map_s - proto_map_t) ** 2+eps))
    return proto_loss,dist

if __name__ == '__main__':
    feature_s=torch.zeros(8,1,224,224)
    feature_t = torch.zeros(8, 1, 224, 224)
    label = torch.randint(0, 2, (8, 1, 224, 224))
    cal_proto_loss(feature_s,feature_t,label)