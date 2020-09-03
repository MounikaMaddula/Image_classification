
import torch
from torch.utils.data import DataLoader 
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn 
from torchvision import transforms, models
import os
import numpy as np 

#importing custom modules
from data_loader import RandomRotation, Monkey_Dataset
from model import Conv_Block, FC_Block, VGGNet
from train import model_train, model_validation

train_dataset = Monkey_Dataset(root_folder = '../monkey_challenge', mode = 'training')
val_dataset = Monkey_Dataset(root_folder = '../monkey_challenge', mode = 'validation')

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle = True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size = 4, shuffle = True, num_workers = 4)

#model = VGGNet(in_channels = 3, out_classes = 11)

pretrained = ['VGG19','resnet18','alexnet','squeezenet',  \
               'densenet','inception','googlenet'][-1]

if pretrained == 'alexnet' :
    model = models.alexnet(pretrained=True)
    
    #Freeze convolution layers
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 11) 

if pretrained == 'inception':
    #Using pre-trained model
    print ('Pretrained Inception')    
    model = models.inception_v3(pretrained=True)

    #Freeze convolution layers
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 11) 


if pretrained == 'squeezenet':
    #Using pre-trained model
    print ('Pretrained Squeezenet')
    model = models.squeezenet1_0(pretrained=True)

    #Freeze convolution layers
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, 11,kernel_size=(1, 1), stride=(1, 1))

if pretrained == 'densenet':
    print ('Pretrained Densenet')
    #Using pre-trained model
    model = models.densenet161(pretrained=True)

    #Freeze convolution layers
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Linear(model.classifier.in_features, 11)


if pretrained == 'googlenet':
    print ('Pretrained Googlenet')
    #Using pre-trained model
    model = models.googlenet(pretrained=True)

    #Freeze convolution layers
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 11)
    

if pretrained == 'VGG19':
    #Using pre-trained model
    model = models.vgg19(pretrained=True)

    #Freeze convolution layers
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 11) 

#Using pretrained resent
if pretrained == 'resnet18' :
    model = models.resnet18(pretrained=True)

    #Freeze convolution layers
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 11)

criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= 0.001,betas=(0, 0.99))
scheduler = lr_scheduler.StepLR(optimizer, step_size = 3, gamma=0.9)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criteria.to(device)

model_train(model, scheduler, optimizer, val_interval = 2, start_epoch = 0, num_epochs = 100,  \
                train_dataloader = train_dataloader, val_dataloader = val_dataloader,  \
                criteria = criteria, out_dir = 'resnet_chkpt', device = device)