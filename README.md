# ARSeg
Reposity of **"Towards Robust Medical Image Referring Segmentation with Incomplete Textual Prompts"**
![image](https://github.com/w7jie/ARSeg/blob/main/img/ARSeg.png)

# Requirement
Python = 3.7 and using the following command:
```
pip install -r requirements.txt
```
# Usage
## 1.Data preparation
### 1.1 QaTa-COV19 and MosMedData+ datasets
You can download the original datasets and the text annotation in [LViT](https://github.com/HUANGLIZI/LViT).
### 1.2 Process dataset
Before training, you need to run following command to split the text annotation into three attribute. Then you need to replace original one with corresponding splited one.
```
python process_data.py
```
## 2.training
You can train your own model by the following command:
```
python train.py
```
## 3.Evaluation
You can edit Config.py and run the following command to get Dice and IoU scores and the visualization results.
```
python test.py
```
# Citation
If you find ARSeg is useful for your research or application, please cite our paper and give us a star.
```
@inproceedings{wang2025towards,
  title={Towards Robust Medical Image Referring Segmentation with Incomplete Textual Prompts},
  author={Wang, Qijie and Lin, Xian and Yan, Zengqiang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={636--646},
  year={2025},
  organization={Springer}
}
```
