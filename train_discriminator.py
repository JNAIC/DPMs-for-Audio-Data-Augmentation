#Judge Model Training
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models as models
import numpy as np
import pandas as pd
from utils import US8K
from timm.models.xception import xception
model=xception(num_classes=10,pretrained=False).cuda()
loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001)
train_dataset=US8K(train=True,transform_size=128,all_in=True)
print(len(train_dataset.data))
test_dataset=US8K(train=False,transform_size=128)
epoches=500
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
for epoch in range(epoches):
    model.train()
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=100,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=100,shuffle=True)
    accuracy=0
    losses_all=0
    all_have=0
    for data,label in train_loader:
        data=data.cuda()
        label=label.cuda()
        output=model(data)
        loss=loss_fn(output,label)
        accuracy+=torch.sum(torch.argmax(output,dim=1)==torch.argmax(label,dim=1))
        all_have+=data.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_all+=loss.item()
    print('epoch:{},train_loss:{},train_accuracy:{}'.format(epoch,losses_all/all_have,accuracy/all_have)) 

    model.eval()
    all_have=0

    accuracy=0
    losses_all=0
    for data,label in test_loader:
        data=data.cuda()
        label=label.cuda()
        output=model(data)
        loss=loss_fn(output,label)
        all_have+=data.shape[0]
        accuracy+=torch.sum(torch.argmax(output,dim=1)==torch.argmax(label,dim=1))
        losses_all+=loss.item()
    print('epoch:{},test_loss:{},test_accuracy:{}'.format(epoch,losses_all/all_have,accuracy/all_have))
torch.save(model,"judge.pt")
