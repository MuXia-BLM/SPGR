#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MuXiaRepair
@File    ：ACASXU_Repir
@Author  ：慕夏
@Date    ：2025/6/25
'''

from ACASXU_Repair_FT import SPGR_REPAIR_FT
import os
import logging
import scipy.io as sio
import multiprocessing
from ACASXU_Repair_List import *
import argparse
import numpy as np

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    savepath = './logs'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    # Creating and Configuring Logger
    logger = logging.getLogger()
    Log_Format = logging.Formatter('%(levelname)s %(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('../logs/neural_network_repair.log', 'w+')
    file_handler.setFormatter(Log_Format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(Log_Format)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=int, default=19)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--min_source_layer', type=int, default=0)
    parser.add_argument('--repair_neurons_num', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_iterations', type=int, default=10)
    args = parser.parse_args()

    networks = repair_list
    num_processors = multiprocessing.cpu_count()
    alpha = args.alpha
    lr = args.lr
    NC = 0.0000
    RSR = 0.0000
    GEN = 0.0000
    DD = 0.0000
    Time = 0.0000

    NC_list = []
    RSR_list = []
    GEN_list = []
    DD_list = []
    Time_list = []
    min_source_layer = 0

    zero = [21,22,23,24,26,29,31,32,34,35,36,38,44,47,49,51,52,53,54,55]
    one = [19,21,28,39,41,43,45,48,56]
    two = [25,27,37,46,57,58,59]

    for n in range(len(networks)):

        item = networks[n]
        i, j = item[0][0], item[0][1]

        if i * 10 + j != args.network:
            continue

        logging.info(f'Neural Network {i} {j}, alpha = {alpha}')
        properties_repair = item[1]

        nn_path = "../networks/nnv_format/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000"
        model = sio.loadmat(nn_path)
        network = i * 10 + j
        spgr_rp = SPGR_REPAIR_FT(network,model, properties_repair, nn_path, safe_num=10000, unsafe_num=1000, neg_num=10000, pos_num=1000,
                       repair_num=args.repair_neurons_num, alpha=alpha, max_iterations = args.max_iterations)

        spgr_rp.get_data(N=i * 10 + j, sample_data=False)

        if network in zero:
            min_source_layer = 0
        if network in one:
            min_source_layer = 1
        if network in two:
            min_source_layer = 2

        NC, RSR, GEN, DD, Time = spgr_rp.solve_safety(alpha=0.1, lambda1=0.9, lambda2=0.1, min_source_layer=min_source_layer)
        NC_list.append(NC)
        RSR_list.append(RSR)
        GEN_list.append(GEN)
        DD_list.append(DD)
        Time_list.append(Time)
        logging.info('\n****************************************************************\n')

    print("平均NC：",np.mean(NC_list))
    print("平均RSR：",np.mean(RSR_list))
    print("平均GEN：",np.mean(GEN_list))
    print("平均DD：",np.mean(DD_list))
    print("平均Time：",np.mean(Time_list))



