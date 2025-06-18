import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init(self):
        super(Attention,self).__init__()

    def forward(self,query,key,value):
        d = torch.tensor(512.0)
        attn = torch.matmul(query,key.transpose(-1, -2)) / torch.sqrt(d)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn,value)
        return out

class MLP(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim):
        super(MLP, self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim,out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self,x):
        return self.layer(x)

class SelfAttention(nn.Module):
    def __init__(self,dim,num_heads=8,mlp_ratio=4):
        super(SelfAttention,self).__init__()
        self.num_heads = num_heads
        self.qkv0 = nn.Linear(dim, dim * 3)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_cal = Attention()
        self.dropout = nn.Dropout(0.1)
        self.mlp = MLP(dim, dim * mlp_ratio, dim)

    def forward(self, x):
        B, N, C = x.shape
        x1 = self.layer_norm1(x)
        qkv = self.qkv0(x1)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # B,N,3,num_heads,head_dims
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3,B,num_heads,N,head_dims
        q, k, v = qkv[0], qkv[1], qkv[2]  # B,num_heads,N,head_dims
        attn = self.attn_cal(q, k, v)  # B,num_heads,N,head_dims
        attn = attn.transpose(1, 2)  # B,N,num_heads,num_dims
        attn = attn.reshape(B, N, C)
        attn_out = self.dropout(self.proj(attn))
        out = x + attn_out
        mlp_out = self.dropout(self.mlp(self.layer_norm1(out)))
        out = out + mlp_out
        return out

class CrossAttention(nn.Module):
    def __init__(self,dim,num_heads=8,mlp_ratio=4):
        super(CrossAttention,self).__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(0.1)
        self.attn_cal = Attention()
        self.proj = nn.Linear(dim, dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dim)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.q=nn.Linear(dim,dim)
        self.kv=nn.Linear(dim,dim)

    def forward(self, X_main, X):#X_main做q,X做k,v
        B, NM, C=X_main.shape
        _, NX, _=X.shape
        X_main=X_main.cuda()
        X=X.cuda()
        x_main=self.layer_norm1(X_main)
        x=self.layer_norm1(X)
        x_main_q=self.q(x_main)
        x_kv=self.kv(x)
        x_main_q=x_main_q.reshape(B,NM,self.num_heads,C//self.num_heads)
        x_kv=x_kv.reshape(B,NX,self.num_heads,C//self.num_heads)
        x_main_q=x_main_q.permute(0, 2, 1, 3)#B,num_heads,NM,head_dims
        x_kv=x_kv.permute(0, 2, 1, 3)#B,num_heads,NX,head_dims
        attn=self.attn_cal(x_main_q,x_kv,x_kv)#B,num_heads,NM,head_dims
        attn=attn.transpose(1, 2)#B,NM,num_heads,head_dims
        attn = attn.reshape(B,NM,C)
        attn_out = self.dropout(self.proj(attn))
        out = X_main + attn_out
        mlp_out = self.dropout(self.mlp(self.layer_norm1(out)))
        out = out + mlp_out
        return out
