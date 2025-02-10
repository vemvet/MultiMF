# -*- coding： utf-8 -*-
'''
@Time: 2022/3/21 17:35
@Author:YilanZhang
@Filename:models.py
@Software:PyCharm
@Email:zhangyilan@buaa.edu.cn
'''

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from cnns.swin_transformer.build import build_model
from cnns.swin_transformer.config import _C as MC
from cnns.swin_transformer.utils import load_pretrained
from cnns.convenext.convnext import convnext_tiny



class MLP(nn.Module):
    '''
    MLP that is used for the subnetwork of metadata.
    '''
    def __init__(self,in_size,hidden_size,out_size,dropout=0.0):
        '''
        :param in_size: input dimension
        :param hidden_size: hidden layer dimension
        :param out_size: output dimension
        '''
        super(MLP,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size,hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,out_size),
            nn.BatchNorm1d(out_size),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        return x



class MetaSubNet(nn.Module):
    '''
    The subnetwork that is used for metadata.
    Maybe the subnetwork that is not needed in the task
    '''

    def __init__(self, in_size, hidden_size, out_size, dropout=0.2,):
        '''
        :param in_size: input dimension
        :param hidden_size: hidden layer dimension
        :param out_size: output dimension
        :param dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(MetaSubNet, self).__init__()
        self.rnn = MLP(in_size,hidden_size,out_size,dropout=dropout)

    def forward(self, x):
        '''
        :param x: tensor of shape (batch_size, sequence_len, in_size)
        :return: tensor of shape (batch_size,out_size)
        '''

        meta_output = self.rnn(x)
        return meta_output



class FeatureExtractor(nn.Module):
    def __init__(self,submodule,extracted_layers):
        super(FeatureExtractor,self).__init__()
        self.submodule=submodule
        self.extracted_layers=extracted_layers

    def forward(self, x):
        outputs=[]
        for name,module in self.submodule._modules.items():
            if name == "fc" : x=torch.flatten(x,1)
            x=module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


class ImgSubNet(nn.Module):
    '''
        The subnetwork that is used for image data
    '''

    def __init__(self,out_size,dropout,args,pretrained=True):
        '''
        :param dropout: dropout probability
        :param pretrained: whether use transfer learning for the task
        '''
        super(ImgSubNet, self).__init__()
        if args['models'] == 'resnet18':
            self.subnet = models.resnet18(pretrained=pretrained)
            self.fc_node = 512#*4
        if args['models'] == 'resnet50':
            self.subnet = models.resnet50(pretrained=pretrained)
            self.fc_node = 512*4

        elif args['models'] == 'regnet':
            self.subnet = models.regnet_y_400mf(pretrained=pretrained)
            self.fc_node = 440#400#912#440 #1.6G888

        elif args['models'] == 'efficientnet':
            self.subnet = models.efficientnet_b0(pretrained=pretrained)
            self.fc_node = 1280#1536

        elif args['models'] == 'vit':
            self.subnet = timm.create_model('vit_small_patch16_224',pretrained=True)
            self.fc_node = 1000
        elif args['models'] == 'swin':
            self.subnet = build_model(MC)
            load_pretrained(MC, self.subnet)
            self.fc_node = 1000

        elif args['models'] == 'resnext':
            self.subnet = models.resnext50_32x4d(pretrained=pretrained)
            self.fc_node = 512*4

        elif args['models'] =='inception':
            self.subnet = models.inception_v3(pretrained=pretrained)
            self.subnet.aux_logits = False
            self.fc_node = 1000

        elif args['models'] == 'densenet':
            self.subnet = models.densenet121(pretrained=pretrained)
            self.fc_node = 1000
        elif args['models'] == 'convenext':
            self.subnet= convnext_tiny(pretrained=pretrained,in_22k=True,num_classes=21841)
            self.fc_node = 768

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.fc_node,out_size)

        if args['models'] == 'vit' or args['models'] == 'swin'or args['models'] == 'densenet':
            self.model = self.subnet
        elif args['models'] == 'inception':
            self.model = self.subnet
            self.model.aux_logits = False
        elif args['models'] == 'convenext':
            self.subnet.head = nn.Sequential()
            self.model = self.subnet
        else:
            self.model = torch.nn.Sequential(*(list(self.subnet.children())[:-1]))



    def forward(self,x):
        x = self.model(x)
        # x = x.view(x.shape[0],-1)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        img_output = self.linear(x)

        return img_output

class Concate_Model(nn.Module):
    '''
    Concate Multimodal Fusion
    '''
    def __init__(self,in_size,hidden_size,out_size,dropouts,args,num_classes=6):
        '''
        :param meta_output: output of MetaSubNet for metadata
        :param img_output1: output of ImgSubNet for clinical image
        :param img_output2: output of ImgSubNet for dermoscopy image
        :param out_size: output demension of SubNets
        :param num_classes: the number of class in the task
        '''
        super(Concate_Model, self).__init__()

        #demension are secified in the order of metadata, clinical image and dermoscopy image
        self.meta_in = in_size[0]
        self.cli_img_in = in_size[1] #224 in resnet18
        self.der_img_in = in_size[2]  #224 in resnet18, maybe we will use efficientNet in the future

        self.meta_hidden = hidden_size[0]

        self.meta_out = out_size[0]
        self.cli_img_out = out_size[1]
        self.der_img_out = out_size[2]

        self.meta_prob = dropouts[0]
        self.cli_img_prob = dropouts[1]
        self.der_img_prob = dropouts[2]

        #define the pre-fusion subnetworks
        self.meta_subnet = MetaSubNet(self.meta_in, self.meta_hidden, self.meta_out, dropout=self.meta_prob)
        self.cli_subnet = ImgSubNet(self.cli_img_out, self.cli_img_prob, args,pretrained=args["pretrained"])
        self.der_subnet = ImgSubNet(self.der_img_out, self.der_img_prob, args,pretrained=args["pretrained"])

        self.num_classes = num_classes

        self.fc_6 = nn.Linear(self.der_img_out+self.cli_img_out+self.meta_out,self.num_classes) # only use one image modality
        self.fc_2 = nn.Linear(self.der_img_out+self.cli_img_out+self.meta_out,2)


    def forward(self, meta_x, cli_x, der_x):
        '''
        :param meta_x: tensor of shape (batch_size, meta_in)
        :param cli_x: tensor of shape (batch_size, cli_img_in)
        :param der_x: tensor of shape (batch_size, der_img_out)
        '''
        meta_h = self.meta_subnet(meta_x)
        cli_h = self.cli_subnet(cli_x)
        der_h = self.der_subnet(der_x)

        #feature fusion
        x = torch.cat([meta_h,cli_h,der_h], dim=-1)

        diag_6 = self.fc_6(x)
        diag_2 = self.fc_2(x)

        return [diag_6,diag_2]

    def criterion(self, logit, truth):
        loss = nn.CrossEntropyLoss()(logit, truth)

        return loss


class Concate_Model_MF(nn.Module):
    '''
    Concate Multimodal Fusion
    '''
    def __init__(self,in_size,hidden_size,out_size,dropouts,num_classes=6,dim=440,pred_dim=128):
        '''
        :param meta_output: output of MetaSubNet for metadata
        :param img_output1: output of ImgSubNet for clinical image
        :param img_output2: output of ImgSubNet for dermoscopy image
        :param out_size: output demension of SubNets
        :param num_classes: the number of class in the task
        '''
        super(Concate_Model_MF, self).__init__()

        #demension are secified in the order of metadata, clinical image and dermoscopy image
        self.meta_in = in_size[0]
        self.cli_img_in = in_size[1] #224 in resnet18
        self.der_img_in = in_size[2]  #224 in resnet18

        self.meta_hidden = hidden_size[0]

        self.meta_out = out_size[0]
        self.cli_img_out = out_size[1]
        self.der_img_out = out_size[2]

        self.meta_prob = dropouts[0]
        self.cli_img_prob = dropouts[1]
        self.der_img_prob = dropouts[2]

        #define the pre-fusion subnetworks
        self.meta_subnet = MetaSubNet(self.meta_in, self.meta_hidden, self.meta_out, dropout=self.meta_prob)

        self.cli_subnet = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1)
        self.cli_subnet = nn.Sequential(*list(self.cli_subnet.children())[:-2])

        self.der_subnet = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1)
        self.der_subnet = nn.Sequential(*list(self.der_subnet.children())[:-2])

        self.num_classes = num_classes

        self.pull = nn.Sequential(
            nn.Conv2d(dim, pred_dim, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(pred_dim, pred_dim, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(pred_dim),
            nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_fusion_6 = nn.Linear(self.meta_out+self.der_img_out+self.cli_img_out,self.num_classes) # only use one image modality
        self.fc_fusion_2 = nn.Linear(self.meta_out+self.der_img_out+self.cli_img_out,2)

        self.fc_cli_6=nn.Linear(self.cli_img_out, self.num_classes)
        self.fc_der_6=nn.Linear(self.der_img_out, self.num_classes)
        self.fc_cli_2=nn.Linear(self.cli_img_out, 2)
        self.fc_der_2=nn.Linear(self.der_img_out, 2)


    def forward(self, meta_x, cli_x, der_x):
        '''
        :param meta_x: tensor of shape (batch_size, meta_in)
        :param cli_x: tensor of shape (batch_size, cli_img_in)
        :param der_x: tensor of shape (batch_size, der_img_out)
        '''
        meta_h = self.meta_subnet(meta_x)
        cli_h = self.cli_subnet(cli_x)
        der_h = self.der_subnet(der_x)

        pt1 = torch.flatten(self.avgpool(self.pull(cli_h)), 1)
        pt2 = torch.flatten(self.avgpool(self.pull(der_h)), 1)

        x = torch.cat([meta_h, pt1, pt2], dim=-1)  # TODO 这里测融和方式
        diag_6 = self.fc_fusion_6(x)
        diag_2 = self.fc_fusion_2(x)

        #单独
        diag_6_cli = self.fc_cli_6(pt1)
        diag_6_der = self.fc_der_6(pt2)
        diag_2_cli = self.fc_cli_2(pt1)
        diag_2_der = self.fc_der_2(pt2)

        return diag_6,diag_2,diag_6_cli,diag_6_der,diag_2_cli,diag_2_der

    def criterion(self, logit, truth):
        loss = nn.CrossEntropyLoss()(logit, truth)

        return loss



