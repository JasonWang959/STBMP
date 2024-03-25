from torch.utils.data import Dataset
import numpy as np
import os
from utils import data_utils

def get_stream(data,view):
    N, T, M =data.shape
    if view=='joint':
        pass
    elif view=='motion':
        motion = np.zeros_like(data)
        motion[:,:-1,:]=data[:,1:,:]-data[:,:-1,:]
        data=motion
    return data

class CMU_Motion3D(Dataset):

    def __init__(self, path_to_data, actions, input_n=10, output_n=10, split=0, data_mean=0, data_std=0, dim_used=0,
                 dct_n=15):

        self.path_to_data = path_to_data
        self.split = split
        actions = data_utils.define_actions_cmu(actions)
        # actions = ['walking']#test用
        if split == 0:
            path_to_data = os.path.join(path_to_data, 'train')
            is_test = False
        else:
            path_to_data = os.path.join(path_to_data, 'test')
            is_test = True
        all_seqs, dim_ignore, dim_use, data_mean, data_std = data_utils.load_data_cmu_3d(path_to_data, actions,
                                                                                         input_n, output_n,
                                                                                         data_std=data_std,
                                                                                         data_mean=data_mean,
                                                                                         is_test=is_test)
        if not is_test:
            dim_used = dim_use

        self.all_seqs = all_seqs
        self.dim_used = dim_used
        all_seqs = all_seqs[:, :, dim_used]

        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)
        input_seqs= all_seqs[:,i_idx,:]


        motion_data = get_stream(all_seqs,'motion')
        motion_data = motion_data[:,i_idx,:]
        t_joint = motion_data # N,T,VC
        s_joint = motion_data.transpose(0,2,1)# N,VC,T
        self.input = input_seqs
        self.input_t = t_joint
        self.input_s = s_joint 

        self.data_mean = data_mean
        self.data_std = data_std
    def __len__(self):
        return np.shape(self.input)[0]

    def __getitem__(self, item):
        return self.input[item], self.input_t[item], self.input_s[item], self.all_seqs[item]
