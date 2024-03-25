import numpy as np
import torch
from torch.autograd import Variable
from utils import data_utils


def sen_loss(outputs, all_seq, dim_used, dct_n):
    """

    :param outputs: N * (seq_len*dim_used_len)
    :param all_seq: N * seq_len * dim_full_len
    :param input_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)
    dim_used = np.array(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    pred_expmap = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                               seq_len).transpose(1, 2)
    targ_expmap = all_seq.clone()[:, :, dim_used]

    loss = torch.mean(torch.sum(torch.abs(pred_expmap - targ_expmap), dim=2).view(-1))
    return loss


def euler_error(outputs, all_seq, input_n, dim_used, dct_n):
    """

    :param outputs:
    :param all_seq:
    :param input_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len, seq_len).transpose(1, 2)
    pred_expmap = all_seq.clone()
    dim_used = np.array(dim_used)
    pred_expmap[:, :, dim_used] = outputs_exp

    pred_expmap = pred_expmap[:, input_n:, :].contiguous().view(-1, dim_full_len)
    targ_expmap = all_seq[:, input_n:, :].clone().contiguous().view(-1, dim_full_len)

    # pred_expmap[:, 0:6] = 0
    # targ_expmap[:, 0:6] = 0
    pred_expmap = pred_expmap.view(-1, 3)
    targ_expmap = targ_expmap.view(-1, 3)

    pred_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(pred_expmap))
    pred_eul = pred_eul.view(-1, dim_full_len)

    targ_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(targ_expmap))
    targ_eul = targ_eul.view(-1, dim_full_len)
    mean_errors = torch.mean(torch.norm(pred_eul - targ_eul, 2, 1))

    return mean_errors


def mpjpe_error(outputs, all_seq, input_n, dim_used, dct_n):
    """

    :param outputs:
    :param all_seq:
    :param input_n:
    :param dim_used:
    :param data_mean:
    :param data_std:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                               seq_len).transpose(1, 2)
    pred_expmap = all_seq.clone()
    dim_used = np.array(dim_used)
    pred_expmap[:, :, dim_used] = outputs_exp
    pred_expmap = pred_expmap[:, input_n:, :].contiguous().view(-1, dim_full_len).clone()
    targ_expmap = all_seq[:, input_n:, :].clone().contiguous().view(-1, dim_full_len)

    pred_expmap[:, 0:6] = 0
    targ_expmap[:, 0:6] = 0

    targ_p3d = data_utils.expmap2xyz_torch(targ_expmap).view(-1, 3)

    pred_p3d = data_utils.expmap2xyz_torch(pred_expmap).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(targ_p3d - pred_p3d, 2, 1))

    return mean_3d_err


def mpjpe_error_cmu(outputs, all_seq, input_n, dim_used, dct_n):
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                               seq_len).transpose(1, 2)
    pred_expmap = all_seq.clone()
    dim_used = np.array(dim_used)
    pred_expmap[:, :, dim_used] = outputs_exp
    pred_expmap = pred_expmap[:, input_n:, :].contiguous().view(-1, dim_full_len)
    targ_expmap = all_seq[:, input_n:, :].clone().contiguous().view(-1, dim_full_len)
    pred_expmap[:, 0:6] = 0
    targ_expmap[:, 0:6] = 0

    targ_p3d = data_utils.expmap2xyz_torch_cmu(targ_expmap).view(-1, 3)
    pred_p3d = data_utils.expmap2xyz_torch_cmu(pred_expmap).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(targ_p3d - pred_p3d, 2, 1))

    return mean_3d_err


def mpjpe_error_p3d(outputs, all_seq, dim_used):#
    """

    :param outputs:b*20*66
    :param all_seq: b*20*96
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape#batch_size，input_n+output_n, (22+10)*3
    dim_used = np.array(dim_used)#66*1
    dim_used_len = len(dim_used)#66
 
    outputs_p3d = outputs                                                            
    pred_3d = outputs_p3d.contiguous().view(-1, dim_used_len).view(-1, 3)
    targ_3d = all_seq[:, :, dim_used].contiguous().view(-1, dim_used_len).view(-1, 3)#b*20*96 →20b*66 → 440b*3

    mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 2, 1))

    return mean_3d_err


def mpjpe_error_3dpw(outputs, all_seq, dct_n, dim_used):
    n, seq_len, dim_full_len = all_seq.data.shape

    _, idct_m = data_utils.get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    outputs_exp = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_full_len - 3,
                                                                                                seq_len).transpose(1,
                                                                                                                   2)
    pred_3d = all_seq.clone()
    pred_3d[:, :, dim_used] = outputs_exp
    pred_3d = pred_3d.contiguous().view(-1, dim_full_len).view(-1, 3)
    targ_3d = all_seq.contiguous().view(-1, dim_full_len).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 2, 1))

    return mean_3d_err

def bone_length_h36(outputs, all_seq, dim_used):
    I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
    J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
    dim_used = np.array(dim_used)#66*1

    n,t,v_c = outputs.shape#32,10,66
    c = 3
    outputs_p3d = all_seq.clone()
    outputs_p3d[:,:,dim_used] = outputs

    outputs_p3d = outputs_p3d.reshape(n,t,-1,c)#32,20,32,3
    targ_3d = all_seq.reshape(n,t,-1,c)#32,20,32,3
    # Calculate bone length loss
    pred_bone_lengths = torch.norm(outputs_p3d[:, :, I, :] - outputs_p3d[:, :, J, :], p=2,dim=-1)#32,20,16
    targ_bone_lengths = torch.norm(targ_3d[:, :, I, :] - targ_3d[:, :, J, :], p=2,dim=-1)#32,20,16
    
    loss = torch.mean(torch.abs(pred_bone_lengths - targ_bone_lengths))
    return loss