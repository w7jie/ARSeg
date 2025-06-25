import pandas as pd
import numpy as np
input_file="test_mosmeddataplus.xlsx"
outpu_file="test_mosmeddataplus_0.xlsx"

df=pd.read_excel(input_file)
def splittext(text):
    sentense=text.replace('\n','').split(',')
    while len(sentense)<3:
        sentense.append('EOF XXX')
    sentense0=sentense[0]
    sentense1=sentense[1]
    sentense2=sentense[2].replace('.','')
    if len(sentense0.split())<5:
        sentense0+=' EOF XXX'*(5-len(sentense0.split()))
    words=sentense0.split()[:5] 
    sentense0=' '.join(words)
    if len(sentense1.split())<5:
        sentense1+=' EOF XXX'*(5-len(sentense1.split()))
    words=sentense1.split()[:5] 
    sentense1=' '.join(words)
    if len(sentense2.split())<10:
        sentense2+=' EOF XXX'*(10-len(sentense2.split()))
    words=sentense2.split()[:10] 
    sentense2=' '.join(words)
    return sentense0,sentense1,sentense2
df['text1'],df['text2'],df['text3']=zip(*df['Description'].apply(splittext))
df_selected=df[['Image', 'text1', 'text2', 'text3']]
df_selected.to_excel(outpu_file,index=False)