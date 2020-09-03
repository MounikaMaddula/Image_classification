
import torch.nn as nn
import torch.nn.functional as F 

class Conv_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, stride, padding) :
        super(Conv_Block, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias = False)
        self.bn = nn.BatchNorm2d(num_features = out_channels)

    def forward(self,x):

        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        return out

class FC_Block(nn.Module):

    def __init__(self, in_feat, out_feat):
        super(FC_Block,self).__init__()
        
        self.fc = nn.Linear(in_feat, out_feat, bias = False)
        self.bn = nn.BatchNorm1d(out_feat)

    def forward(self, x):

        out = self.fc(x)
        out = self.bn(out)
        out = F.relu(out)

        return out


class VGGNet(nn.Module):

    def __init__(self, in_channels, out_classes):
        super(VGGNet, self).__init__()

        self.conv1 = Conv_Block(in_channels, out_channels = 64, kernel = 3, stride = 1, padding = 1)
        self.conv2 = Conv_Block(in_channels = 64, out_channels = 128, kernel = 3, stride = 1, padding = 1)
        self.conv3 = Conv_Block(in_channels = 128, out_channels = 256, kernel = 3, stride = 1, padding = 1)
        self.conv4 = Conv_Block(in_channels = 256, out_channels = 256, kernel = 3, stride = 1, padding = 1)
        self.conv5 = Conv_Block(in_channels = 256, out_channels = 512, kernel = 3, stride = 1, padding = 1)
        self.conv6 = Conv_Block(in_channels = 512, out_channels = 512, kernel = 3, stride = 1, padding = 1)
        self.conv7 = Conv_Block(in_channels = 512, out_channels = 512, kernel = 3, stride = 1, padding = 1)
        self.conv8 = Conv_Block(in_channels = 512, out_channels = 512, kernel = 3, stride = 1, padding = 1)
        self.fc1 = FC_Block(in_feat = 25088, out_feat = 4096)
        self.fc2 = FC_Block(in_feat = 4096, out_feat = 4096)
        self.fc3 = nn.Linear(4096, out_classes)

    def forward(self, x):

        out = self.conv1(x)
        out = F.max_pool2d(out,2)
        out = self.conv2(out)
        out = F.max_pool2d(out,2)
        out = self.conv3(out)
        out = self.conv4(out)
        out = F.max_pool2d(out,2)
        out = self.conv5(out)
        out = self.conv6(out)  
        out = F.max_pool2d(out,2)
        out = self.conv7(out)
        out = self.conv8(out) 
        out = F.max_pool2d(out,2)
        #print (out.shape)
        out = out.view(-1, 25088)
        out = self.fc1(out)
        #print (out.shape)
        out = self.fc2(out)
        #print (out.shape)
        out = self.fc3(out)
        #print (out.shape)
        out = F.softmax(out, dim = 1)

        return out 