from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils import data_utils
from matplotlib import pyplot as plt

def get_stream(data,view):
    N, T, M =data.shape
    if view=='joint':
        pass
    elif view=='motion':
        motion = np.zeros_like(data)
        motion[:,:-1,:]=data[:,1:,:]-data[:,:-1,:]
        motion[:,-1,:]= motion[:,-2,:]
        data=motion
    return data

class H36motion3D(Dataset):

    def __init__(self, path_to_data, actions, input_n=20, output_n=10, split=0, sample_rate=2):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = path_to_data
        self.split = split

        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])
        acts = data_utils.define_actions(actions)

        # subs = np.array([[1], [5], [11]])
        # acts = ['walking']

        subjs = subs[split]
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate, input_n + output_n)

        self.all_seqs = all_seqs
        self.dim_used = dim_used
        all_seqs = all_seqs[:, :, dim_used]

        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)
        input_seqs= all_seqs[:,i_idx,:]

        motion_data = get_stream(all_seqs,'motion')
        motion_data = motion_data[:,i_idx,:]
        t_joint = motion_data 
        s_joint = motion_data.transpose(0,2,1)
        self.input = input_seqs
        self.input_t = t_joint
        self.input_s = s_joint        

    def __len__(self):
        return np.shape(self.input_t)[0]

    def __getitem__(self, item):
        return self.input[item], self.input_t[item], self.input_s[item], self.all_seqs[item]
