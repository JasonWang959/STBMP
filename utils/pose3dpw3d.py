import pickle as pkl
from os import walk
from torch.utils.data import Dataset
import numpy as np
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

class Pose3dPW3D(Dataset):

    def __init__(self, path_to_data, input_n=20, output_n=10, dct_n=15, split=0):
        """

        :param path_to_data:
        :param input_n:
        :param output_n:
        :param dct_n:
        :param split:
        """
        self.path_to_data = path_to_data
        self.split = split
        self.dct_n = dct_n

        # since baselines (http://arxiv.org/abs/1805.00655.pdf and https://arxiv.org/pdf/1705.02445.pdf)
        # use observed 50 frames but our method use 10 past frames in order to make sure all methods are evaluated
        # on same sequences, we first crop the sequence with 50 past frames and then use the last 10 frame as input
        if split == 1:
            their_input_n = 50
        else:
            their_input_n = input_n#
        seq_len = their_input_n + output_n

        if split == 0:
            self.data_path = path_to_data + '/train/'
        elif split == 1:
            self.data_path = path_to_data + '/test/'
        elif split == 2:
            self.data_path = path_to_data + '/validation/'
        all_seqs = []
        files = []
        for (dirpath, dirnames, filenames) in walk(self.data_path):
            files.extend(filenames)
        for f in files:
            with open(self.data_path + f, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                joint_pos = data['jointPositions']
                for i in range(len(joint_pos)):
                    seqs = joint_pos[i]
                    seqs = seqs - seqs[:, 0:3].repeat(24, axis=0).reshape(-1, 72)
                    n_frames = seqs.shape[0]
                    fs = np.arange(0, n_frames - seq_len + 1)
                    fs_sel = fs
                    for j in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + j + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = seqs[fs_sel, :]
                    if len(all_seqs) == 0:
                        all_seqs = seq_sel
                    else:
                        all_seqs = np.concatenate((all_seqs, seq_sel), axis=0)

        self.all_seqs = all_seqs[:, (their_input_n - input_n):, :]
        self.dim_used = np.array(range(3, all_seqs.shape[2]))
        all_seqs = all_seqs[:, (their_input_n - input_n):, 3:]

        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)
        input_seqs= all_seqs[:,i_idx,:]

        motion_data = get_stream(all_seqs,'motion')
        motion_data = motion_data[:,i_idx,:]
        t_joint = motion_data # N,T,VC
        s_joint = motion_data.transpose(0,2,1)# N,VC,T
        self.input = input_seqs
        self.input_t = input_seqs#t_joint
        self.input_s = input_seqs.transpose(0,2,1)#s_joint  

    def __len__(self):
        return np.shape(self.input)[0]

    def __getitem__(self, item):
        return self.input[item]*1000, self.input_t[item]*1000, self.input_s[item]*1000, self.all_seqs[item]*1000

