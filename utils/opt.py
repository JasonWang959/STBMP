#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        # self.parser.add_argument('--data_dir', type=str, default='***/h3.6m/dataset', help='path to H36M dataset')
        # self.parser.add_argument('--data_dir_3dpw', type=str, default='***/sequenceFiles/', help='path to 3DPW dataset')#
        # self.parser.add_argument('--data_dir_cmu', type=str, default='***/', help='path to CMU dataset')
        self.parser.add_argument('--data_dir', type=str, default='/data00/jiexin_wang/datasets/human3.6/h3.6m/dataset', help='path to H36M dataset')
        self.parser.add_argument('--data_dir_3dpw', type=str, default='/data00/jiexin_wang/datasets/3DPW/sequenceFiles/', help='path to 3DPW dataset')#
        self.parser.add_argument('--data_dir_cmu', type=str, default='/data00/jiexin_wang/datasets/CMUMocap/', help='path to CMU dataset')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')

        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--t_input_size', type=int, default=69, help='# layers in linear model, the number of joint used in model')
        self.parser.add_argument('--s_input_size', type=int, default=40, help='# layers in linear model, and s_input_size=input_n + output_n')
        self.parser.add_argument('--kernel_size', type=int, default=2, help='# layers in linear model')
        self.parser.add_argument('--stride', type=int, default=1, help='# layers in linear model')
        self.parser.add_argument('--padding', type=int, default=1, help='# layers in linear model')
        self.parser.add_argument('--factor', type=int, default=2, help='# layers in linear model')
        self.parser.add_argument('--hidden_size', type=int, default=512, help='# layers in linear model')
        self.parser.add_argument('--num_layer', type=int, default=1, help='# layers in linear model')
        self.parser.add_argument('--num_head', type=int, default=4, help='# layers in linear model')
        self.parser.add_argument('--granularity', type=int, default=4, help='# layers in linear model')
        self.parser.add_argument('--encoder', type=str, default="GCN", help='# layers in linear model')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--lr', type=float, default=1.0e-3)#学习率
        self.parser.add_argument('--lr_decay', type=int, default=2, help='every lr_decay epoch do lr decay')#
        self.parser.add_argument('--lr_gamma', type=float, default=0.96)
        self.parser.add_argument('--input_n', type=int, default=10, help='observed seq length')
        self.parser.add_argument('--output_n', type=int, default=30, help='future seq length')
        self.parser.add_argument('--dct_n', type=int, default=40, help='number of DCT coeff. preserved for 3D')
        self.parser.add_argument('--actions', type=str, default='all', help='path to save checkpoint')
        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--dropout', type=float, default=0.3, help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--train_batch', type=int, default=32)
        self.parser.add_argument('--test_batch', type=int, default=128)
        self.parser.add_argument('--job', type=int, default=10, help='subprocesses to use for data loading')
        self.parser.add_argument('--is_load', dest='is_load', action='store_true', help='wether to load existing model')
        self.parser.add_argument('--sample_rate', type=int, default=2, help='frame sampling rate')
        self.parser.add_argument('--is_norm_dct', dest='is_norm_dct', action='store_true', help='whether to normalize the dct coeff')
        self.parser.add_argument('--is_norm', dest='is_norm', action='store_true', help='whether to normalize the angles/3d coordinates')

        self.parser.set_defaults(is_load=False)
        # self.parser.set_defaults(is_load=True)

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        # do some pre-check
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        self.opt.ckpt = ckpt
        self._print()
        return self.opt
