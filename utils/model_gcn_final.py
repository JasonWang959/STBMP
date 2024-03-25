from __future__ import absolute_import
from __future__ import print_function

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .attentions import *
import numpy as np
from utils import data_utils
from scipy.spatial.distance import mahalanobis

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

        
class GC_Block(nn.Module):
    def __init__(self, channal, p_dropout, bias=True, node_n=22, seq_len = 20):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = channal
        self.out_features = channal

        self.gc1 = GraphConvolution(channal, channal, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(channal*node_n)

        self.gc2 = GraphConvolution(channal, channal, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(channal*node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):

        y = self.gc1(x)
        b, c, n = y.shape
        y = y.view(b, -1).contiguous()
        y = self.bn1(y).view(b, c, n).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, c, n = y.shape
        y = y.view(b, -1).contiguous()
        y = self.bn2(y).view(b, c, n).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN_encoder(nn.Module):
    def __init__(self, in_channal, out_channal, node_n=22, seq_len=20, p_dropout=0.3, num_stage=4):
        """
        in_channal: hidden_size
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN_encoder, self).__init__()
        self.num_stage = num_stage
        self.gc1 = GraphConvolution(in_channal,out_channal, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(out_channal*node_n)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(channal=out_channal, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)
        self.gc7 = GraphConvolution(out_channal, out_channal, node_n=node_n)
        self.bn2 = nn.BatchNorm1d(out_channal*node_n)
        self.reshape_conv = GraphConvolution(in_channal, out_channal, node_n=node_n)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, c, n = y.shape
        y = y.view(b, -1).contiguous()
        y = self.bn1(y).view(b, c, n).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        #print("y:{} self.reshape_conv(x):{}".format(y.shape,(self.reshape_conv(x)).shape))
        return y + self.reshape_conv(x)

def kl_divergence(p, q):
    p = F.softmax(p / 0.1, dim=-1)
    q = F.softmax(q / 0.1, dim=-1)
    kl_div = F.kl_div(p.log(), q, reduction='batchmean')
    return kl_div

def batch_norm(frames):
    """
        frames: Tensor, [batch_size, num_frames, frame_dim]
    """
    batch_size, num_frames, frame_dim = frames.shape
    frames = frames.reshape(1, -1, frame_dim)

    mean = torch.mean(frames, dim=1, keepdim=True)
    std = torch.std(frames, dim=1, keepdim=True)
    normalized_frames = (frames - mean) / std

    normalized_frames = normalized_frames.reshape(batch_size, num_frames, frame_dim)
    return normalized_frames

def normalize_frames(frames):
    """
        frames: ndarray, [batch_size, num_frames, frame_dim]
    """
    mean = frames.mean(axis=(0, 1), keepdims=True)
    std = frames.std(axis=(0, 1), keepdims=True)
    normalized_frames = (frames - mean) / std
    return normalized_frames

def calculate_kl_divergence(frame1, frame2):#y_s,y_t
    """
        frame1: [batch_size, num_frames, frame_dim]
        frame2: [batch_size, num_frames, frame_dim]
    """
    frame1 = normalize_frames(frame1)
    frame2 = normalize_frames(frame2)
    log_p = torch.nn.functional.log_softmax(frame1, dim=-1)
    log_q = torch.nn.functional.softmax(frame2, dim=-1)

    kl_div = torch.nn.functional.kl_div(log_p, log_q, reduction='batchmean').sum(dim=-1)
    return kl_div

class STBMP(nn.Module):
    def __init__(
        self,
        t_input_size: int, # the input size of temporal domain data, e.g. 66
        s_input_size: int, # the input size of spatial domain data, e.g. 20
        dct_used: int,
        kernel_size: int, #2
        stride: int,  #1
        padding: int, #1
        factor: int, #2
        hidden_size: int, # the hidden size of STMM model, e.g. 512
        num_head: int, # the number of heads in multi-head attention, e.g. 4
        num_layer: int, # the number of layers in STMM model, e.g. 1
        granularity: int, # the granularity level of the graph convolutional network, e.g. 4
        encoder: str, # the type of encoder, either GRU, LSTM, Transformer, or GCN
        p_dropout: float, # the dropout probability, e.g. 0.1
        inputsize: int, # the size of input data, e.g. 10
        outputsize: int, # the size of output data, e.g. 10 or 15
    ):
        super(STBMP, self).__init__()
        self.dct_used = dct_used
        self.d_model  = hidden_size 
        self.granularity = granularity
        self.encoder = encoder
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.s_input_size = s_input_size
        self.t_input_size = t_input_size
        # temproal and spatial branch embedding layers
        self.t_embedding = nn.Sequential(
                            nn.Linear(t_input_size, hidden_size),
                            nn.LayerNorm(hidden_size),
                            nn.ReLU(True),
                            nn.Linear(hidden_size, hidden_size),
        ) 
        self.s_embedding = nn.Sequential(
                            nn.Linear(self.dct_used, hidden_size),
                            nn.LayerNorm(hidden_size),
                            nn.ReLU(True),
                            nn.Linear(hidden_size, hidden_size),
        )

        # seq2seq encoders
        if  encoder=="GCN":
            self.t_encoder = GCN_encoder(in_channal=self.d_model, out_channal=self.d_model,
                                                 node_n=s_input_size,
                                                 p_dropout=p_dropout,
                                                 num_stage=self.granularity)
            self.s_encoder = GCN_encoder(in_channal=self.d_model, out_channal=self.d_model,
                                                 node_n=t_input_size,
                                                 p_dropout=p_dropout,
                                                 num_stage=self.granularity)
        else:
            raise ValueError("Unknown encoder!")
        # decoder
        self.t_decoder = GraphConvolution(self.d_model, t_input_size, node_n= s_input_size)
        self.s_decoder = GraphConvolution(self.d_model, self.dct_used, node_n=t_input_size)

        # others 
        self.dct_m_in, _ = data_utils.get_dct_matrix(self.inputsize+self.outputsize)
        self.dct_m_out, _ = data_utils.get_dct_matrix(self.inputsize+self.outputsize)
        self.dct_m_in = torch.from_numpy(self.dct_m_in).float().cuda()
        self.dct_m_out = torch.from_numpy(self.dct_m_out).float().cuda()

        self.loss_feat = nn.SmoothL1Loss(beta=2.0)

    def forward(self,x, x_t, x_s):
        if self.encoder=="GRU" or self.encoder=="LSTM":
            self.t_encoder.flatten_parameters()
            self.s_encoder.flatten_parameters()
        ### DCT
        xs = x_s
        xs = xs.reshape(-1, self.inputsize+self.outputsize)
        xs = xs.transpose(1,0)
        xs = torch.matmul(self.dct_m_in[0:self.dct_used, :], xs)
        xs = xs.transpose(1,0).reshape([-1, self.t_input_size, self.dct_used,])
        
        xt = self.t_embedding(x_t) # temporal domain 20,66--20,512 
        xs = self.s_embedding(xs) # spatial domain 66,20--66,512

        if self.encoder=="GRU" or self.encoder=="LSTM":
            vt,_ = self.t_encoder(xt)#32,20,512--32,20,512
            vp, _ = self.s_encoder(xs)#32,66,512--32,66,512
        elif self.encoder=="GCN":
            vt = self.t_encoder(xt) 
            vp = self.s_encoder(xs)      
        else:
            vt = self.t_encoder(xt)
            vp = self.s_encoder(xs)
        y_t = self.t_decoder(vt)#32,20,66

        ### IDCT
        y_s = self.s_decoder(vp)#32,66,15
        y_s = y_s.view(-1, self.dct_used).transpose(0, 1)#32,66,15-->32*66,15-->15,32*66
        y_s = torch.matmul(self.dct_m_out[:, 0:self.dct_used], y_s).transpose(0, 1).contiguous().view(-1, self.t_input_size, self.inputsize+self.outputsize).transpose(1, 2)#32,20,66

        n,t,v_c = y_s.shape
        y_s_xyz = y_s#32,20,66
        y_t_xyz = y_t

        loss = 0.0
        loss = loss + calculate_kl_divergence(y_s_xyz[:,:,:],y_t_xyz[:,:,:])#

        y_s_xyz = y_s_xyz.transpose(2,1)
        y_t_xyz = y_t_xyz.transpose(2,1)
        loss = loss + calculate_kl_divergence(y_t_xyz[:,:,:],y_s_xyz[:,:,:])

        return y_s+x, y_t + x ,0.5*y_s+0.5*y_t + x, loss