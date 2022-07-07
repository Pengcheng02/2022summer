# 此处用已经训练好的模型检验
from train import *
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter



def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_test = pd.read_csv('./data/covid.test.csv').values
    test_dataset=COVID19Dataset(x_test)

    # Select features
    test_loader = DataLoader(test_dataset, 256, shuffle=False, pin_memory=True)
    fmodel = My_Model(input_dim=117).to(device)
    fmodel.load_state_dict(torch.load('./models/model.ckpt'))
    pred = predict(test_loader,fmodel,device)
    print(pred)
    # result = pd.DataFrame(columns=['predict'])
    # #创建一个空的Dataframe
    # #columns 只有一项是用[]
    # # result =pd.DataFrame(columns=['name'])
    # for p in pred:
    #     result=result.append(pd.DataFrame({'name':[p]}),ignore_index=True)
    # filepath = './result.'
    # sheetname = 'result'
    # if os.path.exists(filepath):
    #     # 然后再实例化ExcelWriter
    #     # 使用with方法打开了文件，生成的文件操作实例在with语句之外是无效的，因为with语句之外文件已经关闭了。无需writer.save()和writer.close()，否则报错
    #     with pd.ExcelWriter(filepath, engine="openpyxl", mode='a') as writer:
    #         # 保存到本地excel
    #         # 接下来还是调用to_excel, 但是第一个参数不再是文件名, 而是上面的writer
    #         result.to_excel(writer, sheet_name=sheetname)
    # else:
    #     result.to_excel(filepath, sheet_name=sheetname, index=False)


