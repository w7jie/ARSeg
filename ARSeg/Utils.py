import pandas as pd
import torch

def read_text(filename):
    df=pd.read_excel(filename)
    text1,text2,text3={},{},{}
    for i in df.index.values:
        t1=df.text1[i]
        t2=df.text2[i]
        t3=df.text3[i]
        text1[df.Image[i]] = t1
        text2[df.Image[i]] = t2
        text3[df.Image[i]] = t3
    return text1,text2,text3
# return 3 dict (key: values)

def process_text_segment(segment, length):
    words = segment
    if len(words.split()) < length:
        words+=' EOFXXX'*(length-len(words.split()))
    return words[:length]

def generate_text_mask(input_probability=None):
    if input_probability is None:
        input_probability = [0.5, 0.5, 0.5]
    input_probability=torch.tensor(input_probability)
    random_vector=torch.rand(3)
    text_mask=(random_vector>input_probability).int()
    return text_mask

def predict_output(input_tensor):#B,1,224,224
    bi_output=((input_tensor>0.5)*255)
    return bi_output