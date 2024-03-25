from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional
import numpy as np
from progress.bar import Bar
import pandas as pd

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.pose3dpw3d import Pose3dPW3D
import utils.model_gcn_final as nnmodel
import utils.data_utils as data_utils

def main(opt):
    random_seed=0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  
    torch.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    is_cuda = torch.cuda.is_available()

    # save option in log
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + '_in{:d}_out{:d}'.format(opt.input_n, opt.output_n)
    checkpoint_dir = '{}_in{:d}_out{:d}'.format(os.path.split(os.path.split(opt.data_dir)[0])[1], opt.input_n, opt.output_n)

    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    sample_rate = opt.sample_rate#
    # data loading
    print(">>> loading data")
    train_dataset = Pose3dPW3D(path_to_data=opt.data_dir_3dpw, input_n=input_n, output_n=output_n, split=0)
    dim_used = train_dataset.dim_used
    test_dataset = Pose3dPW3D(path_to_data=opt.data_dir_3dpw, input_n=input_n, output_n=output_n, split=1)
    val_dataset = Pose3dPW3D(path_to_data=opt.data_dir_3dpw, input_n=input_n, output_n=output_n, split=2)
    # load dadasets for training
    train_loader = DataLoader(dataset=train_dataset,batch_size=opt.train_batch,
        shuffle=True,num_workers=opt.job,pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=opt.test_batch,
        shuffle=False,num_workers=opt.job,pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset,batch_size=opt.test_batch,
        shuffle=False,num_workers=opt.job,pin_memory=True)
    print(">>> data loaded !")

    ### Model
    model = nnmodel.STBMP(t_input_size=opt.t_input_size, s_input_size=opt.s_input_size, dct_used=opt.dct_n,
                 kernel_size=opt.kernel_size, stride=opt.stride, padding=opt.padding, factor=opt.factor,
                 hidden_size=opt.hidden_size, num_head=opt.num_head, num_layer=opt.num_layer,
                 granularity=opt.granularity,
                 encoder = opt.encoder,
                 p_dropout = opt.dropout,
                 inputsize = opt.input_n, outputsize = opt.output_n)
                 
    if is_cuda:
        model.cuda()
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if opt.is_load:
        model_path_len = 'checkpoint/test/ckpt_main_last.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location='cpu')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    for epoch in range(start_epoch, opt.epochs):
        if (epoch + 1) % opt.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)

        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        lr_now, t_3d = train(train_loader, model, optimizer, lr_now=lr_now, is_cuda=is_cuda,dim_used=train_dataset.dim_used)
        ret_log = np.append(ret_log, [lr_now, t_3d * 1000])
        head = np.append(head, ['lr', 't_3d'])

        v_3d = val(val_loader, model,is_cuda=is_cuda,dim_used=dim_used)

        ret_log = np.append(ret_log, v_3d * 1000)
        head = np.append(head, ['v_3d'])

        test_3d = test(test_loader,  model, input_n=input_n, output_n=output_n, is_cuda=is_cuda,dim_used=train_dataset.dim_used)
        print("Average: {} | {}".format(test_3d, test_3d.mean()))

        ret_log = np.append(ret_log, test_3d * 1000)
        if output_n == 15:
            head = np.append(head, ['1003d', '2003d', '3003d', '4003d', '5003d'])
        elif output_n == 30:
            head = np.append(head, ['1003d', '2003d', '3003d', '4003d', '5003d', '6003d', '7003d', '8003d', '9003d','10003d'])

        # update log file
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            if not os.path.exists(opt.ckpt + '/' + checkpoint_dir):
              os.makedirs(opt.ckpt + '/' + checkpoint_dir)
            df.to_csv(opt.ckpt + '/' + checkpoint_dir + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(opt.ckpt + '/' + checkpoint_dir + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        # save ckpt
        is_best = v_3d < err_best
        err_best = min(v_3d, err_best)
        file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,'lr': lr_now,
                         'err': test_3d[0],'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()},
                        ckpt_path=opt.ckpt,is_best=is_best,file_name=file_name)

def train(train_loader, model, optimizer, lr_now=None, is_cuda=False, dim_used=[]):
    t_3d = utils.AccumLoss()
    model.train()
    st = time.time()
    for i, (input, input_t, input_s, all_seq) in enumerate(train_loader):
        batch_size = input.shape[0]
        if batch_size == 1:
            break
        bt = time.time()

        if is_cuda:
            input = input.cuda().float()
            input_t = input_t.cuda().float()
            input_s = input_s.cuda().float()
            all_seq = all_seq.cuda(non_blocking=True).float()
        else:
            input = input.float()
            input_t = input_t.float()
            input_s = input_s.float()
            all_seq = all_seq.float()
        outputs, outputs_t, outputs_ave, mean_3d_err = model(input, input_t, input_s)
        loss_s = loss_funcs.mpjpe_error_p3d(outputs, all_seq, dim_used)
        loss_t = loss_funcs.mpjpe_error_p3d(outputs_t, all_seq, dim_used)
        loss = loss_s + loss_t + 0.1*mean_3d_err

        # calculate loss and backward
        optimizer.zero_grad()
        loss.backward()
        if True:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        n, seq_len, _ = all_seq.data.shape
        t_3d.update(loss.cpu().data.item() * n * seq_len, n * seq_len)
        
        if i%5000==0:
            print('{}/{} | TrainLoss {:.4f}, PointLoss {:4f} batch time {:.4f}s|total time{:.2f}s'\
                       .format(i+1, len(train_loader), t_3d.avg, loss.item(), time.time() - bt, time.time() - st))

    return lr_now, t_3d.avg

def val(train_loader, model, is_cuda=False, dim_used=[]):
    t_3d = utils.AccumLoss()
    model.eval()
    st = time.time()
    with torch.no_grad():
        for i, (input, input_t, input_s, all_seq) in enumerate(train_loader):
            bt = time.time()
            if is_cuda:
                input = input.cuda().float()
                input_t = input_t.cuda().float()
                input_s = input_s.cuda().float()
                all_seq = all_seq.cuda(non_blocking=True).float()   
            else:
                input = input.float()
                input_t = input_t.float()
                input_s = input_s.float()
                all_seq = all_seq.float()   
            outputs,outputs_t, outputs_ave, mean_3d_err = model(input, input_t, input_s)
            n, _, _ = all_seq.data.shape
            m_err = loss_funcs.mpjpe_error_p3d(outputs, all_seq, dim_used)
            # update the training loss
            t_3d.update(m_err.item() * n, n)
            if i%1000==0:
                print('{}/{}|Val set: TrainLoss {:.4f}, batch time {:.4f}s|total time{:.2f}s'\
                           .format(i+1, len(train_loader), t_3d.avg, time.time() - bt, time.time() - st))
    return t_3d.avg

def test(train_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[]):
    N = 0
    if output_n == 15:
        eval_frame = [2, 5, 8, 11, 14]
    elif output_n == 30:
        eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    t_3d = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    with torch.no_grad():
        for i, (input, input_t, input_s, all_seq) in enumerate(train_loader):
            bt = time.time()
    
            if is_cuda:
                input = input.cuda().float()
                input_t = input_t.cuda().float()
                input_s = input_s.cuda().float()
                all_seq = all_seq.cuda(non_blocking=True).float()   
            else:
                input = input.float()
                input_t = input_t.float()
                input_s = input_s.float()
                all_seq = all_seq.float()   
            outputs_3d,outputs_t, outputs_ave, mean_3d_err= model(input, input_t, input_s)
            n, seq_len, dim_full_len = all_seq.data.shape
            dim_used_len = len(dim_used)

            pred_3d = all_seq.clone()
            pred_3d[:, :, dim_used] = outputs_ave
            pred_p3d = pred_3d.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
            targ_p3d = all_seq.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
    
            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                t_3d[k] += torch.mean(torch.norm(
                    targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2,
                    1)).cpu().data.item() * n
            N += n
        
    print('A ErrT: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}, TestError {:.4f}, total time{:.2f}s'\
             .format(float(t_3d[0])/N, float(t_3d[1])/N, float(t_3d[2])/N, float(t_3d[3])/N, float(t_3d[4])/N, 
                     float(t_3d.mean())/N, time.time() - st))
    return t_3d / N

if __name__ == "__main__":
    option = Options().parse()
    main(option)