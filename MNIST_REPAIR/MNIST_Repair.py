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



i, j = 3, 100
nn_path = "../benchmarks/mnist_relu_3_100/mnist_relu_" + str(i) + "_" + str(j) + ".mat"
model = sio.loadmat(nn_path)

rp = ft.SPGR_REPAIR_FT(model,50,0.5,20)
rp.get_data("mnist_relu_3_100")


rp.solve_robustness()

