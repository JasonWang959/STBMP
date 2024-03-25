import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import pandas as pd

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.h36motion3d import H36motion3D
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

    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + '_in{:d}_out{:d}'.format(opt.input_n, opt.output_n)
    checkpoint_dir = '{}_in{:d}_out{:d}'.format(os.path.split(os.path.split(opt.data_dir)[0])[1], opt.input_n, opt.output_n)
    
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    sample_rate = opt.sample_rate

    print(">>> loading data")
    train_dataset = H36motion3D(path_to_data=opt.data_dir, 
                                actions=opt.actions,
                                input_n=input_n, 
                                output_n=output_n,
                                split=0,
                                sample_rate=sample_rate)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=opt.train_batch,
                              shuffle=True,
                              num_workers=opt.job,
                              pin_memory=True)
 
    val_dataset = H36motion3D(path_to_data=opt.data_dir, 
                              actions=opt.actions, 
                              input_n=input_n, 
                              output_n=output_n, 
                              split=2, 
                              sample_rate=sample_rate)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=opt.test_batch,
                            shuffle=False,
                            num_workers=opt.job,
                            pin_memory=True)

    acts = data_utils.define_actions('all')
    test_data = dict()
    for act in acts:
        test_dataset = H36motion3D(path_to_data=opt.data_dir, 
                                   actions=act, 
                                   input_n=input_n, 
                                   output_n=output_n, 
                                   split=1, 
                                   sample_rate=sample_rate,)
        test_data[act] = DataLoader(dataset=test_dataset, 
                                    batch_size=opt.test_batch, 
                                    shuffle=False, 
                                    num_workers=opt.job, 
                                    pin_memory=True)
    
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
        model_path_len = 'checkpoint/test/{}/'.format(checkpoint_dir) + 'ckpt_' + script_name + '_last.pth.tar'
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
        
        ret_log = np.array([start_epoch])
        ret_log = np.append(ret_log, [lr_now, 0.0])

        test_3d_temp = np.array([])
        test_3d_head = np.array([])

        acts_ = ['sitting']
        for act in acts:
            test_l, test_3d = test((act, test_data[act]), model, input_n=input_n, output_n=output_n, is_cuda=is_cuda,
                                   dim_used=train_dataset.dim_used)
            test_3d_temp = np.append(test_3d_temp, test_3d)
            test_3d_head = np.append(test_3d_head, [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
            if output_n > 10:
                test_3d_head = np.append(test_3d_head,
                                         [act + '3d560', act + '3d1000'])
        ret_log = np.append(ret_log, test_3d_temp)
        avg_test3d = ret_log[3:].reshape(15,-1).mean(0)
        ret_log = np.append(ret_log, avg_test3d)
        ret_log = np.append(ret_log, avg_test3d.mean())
        print("Average: {} | {}".format(avg_test3d, avg_test3d.mean()))        

    for epoch in range(start_epoch, opt.epochs):

        if (epoch + 1) % opt.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)

        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        lr_now, t_l = train(train_loader, model, optimizer, lr_now=lr_now, is_cuda=is_cuda,dim_used=train_dataset.dim_used)
        ret_log = np.append(ret_log, [lr_now, t_l])
        head = np.append(head, ['lr', 't_l'])

        v_3d = val(val_loader, model, is_cuda=is_cuda, dim_used=train_dataset.dim_used)

        print("训练集损失、验证集损失：",t_l, v_3d)

        ret_log = np.append(ret_log, [v_3d])
        head = np.append(head, ['v_3d'])

        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        for act in acts:
            test_l, test_3d = test((act, test_data[act]), model, input_n=input_n, output_n=output_n, is_cuda=is_cuda,
                                   dim_used=train_dataset.dim_used)#
            ret_log = np.append(ret_log, test_3d)
            head = np.append(head, [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
            if output_n > 10:
                head = np.append(head, [act + '3d560', act + '3d1000'])
        ret_log = np.append(ret_log, test_3d_temp)
        
        avg_test3d = ret_log[4:].reshape(15,-1).mean(0)
        ret_log = np.append(ret_log, avg_test3d)
        ret_log = np.append(ret_log, avg_test3d.mean())
        print("Average: {} | {}".format(avg_test3d, avg_test3d.mean()))
        head = np.append(head, test_3d_head)
        head = np.append(head, [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
        if output_n > 10:
            head = np.append(head, [act + '3d560', act + '3d1000'])
        head = np.append(head, 'Average')

        # update log file and save checkpoint
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            if not os.path.exists(opt.ckpt + '/' + checkpoint_dir):
              os.makedirs(opt.ckpt + '/' + checkpoint_dir)
            df.to_csv(opt.ckpt + '/' + checkpoint_dir + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(opt.ckpt + '/' + checkpoint_dir + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
                
        #print(os.system('ls ' + opt.ckpt + '/' + checkpoint_dir + '/'))#
        
        if not np.isnan(v_3d):
            is_best = v_3d < err_best
            err_best = min(v_3d, err_best)
            #print("best.pth has updated")
        else:
            is_best = False
        file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'err': test_3d[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=opt.ckpt + '/' + checkpoint_dir,
                        is_best=is_best,
                        file_name=file_name)

def train(train_loader, model, optimizer, lr_now=None, is_cuda=False, dim_used=[]):
    t_l = utils.AccumLoss()
    model.train()
    st = time.time()
    for i, (input, input_t, input_s, all_seq) in enumerate(train_loader):
        batch_size = input_t.shape[0]
        if batch_size == 1:
            continue

        bt = time.time()
        if is_cuda:
            input = input.cuda().float()
            input_t = input_t.cuda().float()
            input_s = input_s.cuda().float()
            all_seq = all_seq.cuda(non_blocking=True).float()
        #print('Shape: t_{} , s_{}  all_{}'.format(input_t.shape,input_s.shape,all_seq.shape))
        outputs, outputs_t,outputs_ave, mean_3d_err = model(input, input_t, input_s)
        loss_s = loss_funcs.mpjpe_error_p3d(outputs, all_seq, dim_used)
        loss_t = loss_funcs.mpjpe_error_p3d(outputs_t, all_seq, dim_used)
        #loss_bone = loss_funcs.bone_length_h36(outputs, all_seq, dim_used)
        loss = loss_s + loss_t + 0.1*mean_3d_err
        #print("iter: ", i ,"total loss:",loss,"  KL loss:",mean_3d_err)
        optimizer.zero_grad()
        loss.backward()
        if True:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        t_l.update(loss.item() * batch_size, batch_size)
        if i%5000==0:
            print('{}/{} |Training set: TrainLoss {:.4f}, PointLoss {:4f} batch time {:.4f}s|total time{:.2f}s'\
                       .format(i+1, len(train_loader), t_l.avg, loss.item(), time.time() - bt, time.time() - st))
    return lr_now,t_l.avg

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
            outputs,outputs_t, outputs_ave,mean_3d_err = model(input, input_t, input_s)
            n, _, _ = all_seq.data.shape
            m_err = loss_funcs.mpjpe_error_p3d(outputs, all_seq, dim_used)
            # update the training loss
            t_3d.update(m_err.item() * n, n)
            if i%1000==0:
                print('{}/{}|Val set: TrainLoss {:.4f}, batch time {:.4f}s|total time{:.2f}s'\
                           .format(i+1, len(train_loader), t_3d.avg, time.time() - bt, time.time() - st))
    return t_3d.avg

def test(act_train_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[]):
    act, train_loader = act_train_loader[0], act_train_loader[1]
    N = 0
    t_l = 0
    if output_n == 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]
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
        
            outputs_3d,outputs_t,outputs_ave, mean_3d_err= model(input, input_t, input_s)
            n, seq_len, dim_full_len = all_seq.data.shape
            dim_used_len = len(dim_used)

            pred_3d = all_seq.clone()
            dim_used = np.array(dim_used)

            joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
            index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
            joint_equal = np.array([13, 19, 22, 13, 27, 30])
            index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

            pred_3d[:, :, dim_used] = outputs_ave
            pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
            pred_p3d = pred_3d.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
            targ_p3d = all_seq.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]

            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                t_3d[k] += torch.mean(torch.norm(targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2, 1)).item() * n 

            N += n

    actname = "{0: <14} |".format(act)
    print('Act: {},  ErrT: {:.3f} {:.3f} {:.3f} {:.3f}, TestError {:.4f}, total time{:.2f}s'\
             .format(actname, 
                     float(t_3d[0])/N, float(t_3d[1])/N, float(t_3d[2])/N, float(t_3d[3])/N, 
                     float(t_3d.mean())/N, time.time() - st))
    return t_l / N, t_3d / N 

if __name__ == "__main__":
    option = Options().parse()
    main(option)