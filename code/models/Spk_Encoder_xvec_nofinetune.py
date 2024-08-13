import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from .WavLM import *

import torch
import torch.nn as nn

class MHFA(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(MHFA, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_v = nn.Parameter(data=torch.ones(13), requires_grad=True)

        self.n_mels     = 768

        p_dropout = 0.1 #0.025

        self.tdnn1 = nn.Conv1d(in_channels=self.n_mels, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=True)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=True)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=True)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=True)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=True)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(1500*2,512)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=True)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(512,512)

        self.attention = nn.Sequential(
            nn.Conv1d(1500,256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256, momentum=0.1, affine=True),
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Softmax(dim=2),
            )



    def forward(self, x):

        # Compute the value in a similar fashion
        x_input = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)
        x_input = x_input.permute(0,2,1)
        #print("Input",x_input.shape)

        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x_input))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))

        eps = 0.0000001
        if self.training:
            shape = x.size()
            noise = torch.FloatTensor(shape)
            noise = noise.to("cuda")
            torch.randn(shape, out=noise)
            x += noise*eps

        w = self.attention(x)
        mu = x * w
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
        stats = torch.cat((mu,sg),1)

        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        x = self.fc2(x)

        #print("Embedding",x.shape)
        #exit(0)

        return x


class spk_extractor(nn.Module):
    def __init__(self,**kwargs):
        super(spk_extractor, self).__init__()
        # checkpoint = torch.load('/mnt/proj3/open-24-5/pengjy_new/WavLM/Pretrained_model/WavLM-Base+.pt')
        print("Pre-trained Model: {}".format(kwargs['pretrained_model_path']))
        checkpoint = torch.load(kwargs['pretrained_model_path'])
        cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(cfg)
        self.loadParameters(checkpoint['model'])
        self.backend = MHFA(head_nb=32)


    def forward(self,wav_and_flag):
        #print("In forward propagation")
        x = wav_and_flag[0]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                cnn_outs, layer_results =  self.model.extract_features(x, output_layer=13)
                layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
                x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)

        #print("Input to backend",x.shape)
        out = self.backend(x)
        return out

    def loadParameters(self, param):

        self_state = self.model.state_dict();
        loaded_state = param

        for name, param in loaded_state.items():
            origname = name;
            

            if name not in self_state:
                # print("%s is not in the model."%origname);
                continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);


def MainModel(**kwargs):
    model = spk_extractor(**kwargs)
    return model
