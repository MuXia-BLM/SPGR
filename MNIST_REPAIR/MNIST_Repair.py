#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MuXiaRepair
@File    ：MNIST_Repair
@Author  ：慕夏
@Date    ：2025/6/26
'''

import numpy as np
import ACASXU_Repair_FT as ft
import scipy.io as sio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--lambda1', type=float, default=0.9)
parser.add_argument('--lambda2', type=float, default=0.1)
args = parser.parse_args()

i, j = 3, 100
nn_path = "../benchmarks/mnist_relu_3_100/mnist_relu_" + str(i) + "_" + str(j) + ".mat"
model = sio.loadmat(nn_path)

rp = ft.SPGR_REPAIR_FT(model,args.alpha,20)
rp.get_data("mnist_relu_3_100")


rp.solve_robustness(beta=args.beta, lambda1=args.lambda1,lambda2=args.lambda2)

