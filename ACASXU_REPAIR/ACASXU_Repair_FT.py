#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MuXiaRepair
@File    ：ACASXU_Repair_FT
@Author  ：慕夏
@Date    ：2025/6/25
'''
from collections import Counter

import pyswarms as ps
from itertools import zip_longest
import h5py
import time
import torch
import numpy as np
from matplotlib.lines import drawStyles
from sklearn.metrics import euclidean_distances
from torch_cluster import nearest
from tqdm import tqdm
from collections import Counter, defaultdict
import os
from include.utils.vnnlib import vnnlib_to_properties
def relu(x):
    return np.maximum(x, 0)


def leaky_relu(x):
    return np.where(x < 0, 0.5 * x, x)


def tanh(x):
    return np.tanh(x)


def elu(x):
    return np.where(x <= 0, 0.5 * (np.exp(x) - 1), x)



class SPGR_REPAIR_FT:

    def __init__(self, network,model, properties_repair, nn_path, safe_num, unsafe_num, neg_num, pos_num, repair_num, alpha,max_iterations,
                 data=None):

        """
        Constructs all the necessary attributes for the Repair object

        Parameters:
            model (Pytorch): A network model
            properties_repair (list): Safety properties and functions to correct unsafe elements.
            data (list): Data for training, validating and testing the network model
            unsafe_num (int): Maximal negative samples from the pre-condition input
            safe_num (int): Maximal positive samples from the neighbourhood
        """
        self.network = network
        self.properties = [item[0] for item in properties_repair]
        self.safe_num = safe_num
        self.unsafe_num = unsafe_num
        self.neg_num = neg_num
        self.pos_num = pos_num
        self.model = model
        self.repair_num = repair_num
        self.alpha = alpha
        self.safe_data = []
        self.unsafe_data = []
        self.pos_data = []
        self.neg_data = []
        self.nn_path = nn_path
        self.max_iterations = max_iterations
        self.repair_neural_num = 1
        self.repair_neural_list = []
        self.threshold = 0.00001
        self.best_pos = 0.0
        self.NC = 0
        self.RSR = 0.0
        self.GEN = 0.0
        self.DD = 0.0
        self.Time = 0.0

    def get_data(self, N=0, sample_data=True):
        # 读取测试安全集
        with h5py.File(f"../benchmarks/acas_N{N}/data/drawndown_test.h5", "r") as f:
            key = "drawndown_test" if "drawndown_test" in f else list(f.keys())[0]
            safe_in = np.array(f[key][:], dtype=np.float32)

        # 读取训练安全集
        with h5py.File(f"../benchmarks/acas_N{N}/data/drawndown.h5", "r") as f:
            key = "drawndown" if "drawndown" in f else list(f.keys())[0]
            pos_in = np.array(f[key][:], dtype=np.float32)

        # 读取测试不安全集
        with h5py.File(f"../benchmarks/acas_N{N}/data/counterexample_test.h5", "r") as f:
            key = "counterexample_test" if "counterexample_test" in f else list(f.keys())[0]
            unsafe_in = np.array(f[key][:], dtype=np.float32)

        # 读取训练不安全集
        with h5py.File(f"../benchmarks/acas_N{N}/data/counterexample.h5", "r") as f:
            key = "counterexample" if "counterexample" in f else list(f.keys())[0]
            neg_in = np.array(f[key][:], dtype=np.float32)


        safe_out = self.get_output(safe_in)
        pos_out = self.get_output(pos_in)
        unsafe_out = self.get_output(unsafe_in)
        neg_out = self.get_output(neg_in)

        self.safe_num = safe_in.shape[0]
        self.pos_num = pos_in.shape[0]
        self.unsafe_num = unsafe_in.shape[0]
        self.neg_num = neg_in.shape[0]

        self.safe_data = [safe_in, safe_out]
        self.unsafe_data = [unsafe_in, unsafe_out]
        self.pos_data = [pos_in, pos_out]
        self.neg_data = [neg_in, neg_out]

    def get_output(self, input):

        X = input.T if input.ndim == 2 and input.shape[1] == 1 else input
        if X.ndim == 1:
            X = X[None, :]
        X = X.astype(np.float32, copy=False)
        num_layers = self.model['W'].shape[1]
        for i in range(num_layers):
            W = self.model['W'][0, i].astype(np.float32, copy=False)  # (out,in)
            b = self.model['b'][0, i].astype(np.float32, copy=False).ravel()
            X = X @ W.T + b
            if i < num_layers - 1:
                np.maximum(X, 0, out=X)
        return X.T if input.ndim == 2 and input.shape[1] == 1 else X

    def purify_data(self, data):
        X = np.asarray(data[0], dtype=np.float32, order='C')
        Y = np.asarray(data[1], dtype=np.float32, order='C')
        keep_mask = np.ones(X.shape[0], dtype=bool)
        for p in self.properties:
            lb, ub = np.asarray(p.lbs, dtype=np.float32), np.asarray(p.ubs, dtype=np.float32)

            in_mask = np.all((X > lb) & (X < ub), axis=1)
            if not in_mask.any():
                continue

            for M_raw, vec_raw in p.unsafe_domains:
                M = np.asarray(M_raw, dtype=np.float32, order='C')
                v = np.asarray(vec_raw, dtype=np.float32, order='C').ravel()
                idx = np.nonzero(in_mask & keep_mask)[0]
                if idx.size == 0:
                    continue
                YY = Y[idx]
                left = (YY @ M.T) + v
                violate = np.all(left <= 0.0, axis=1)
                keep_mask[idx[violate]] = False
        return [X[keep_mask], Y[keep_mask]]

    def model_repair(self, x, r_weight):

        X = x.T if x.ndim == 2 and x.shape[1] == 1 else x
        if X.ndim == 1:
            X = X[None, :]
        X = X.astype(np.float32, copy=False)


        num_layers = self.model['W'].shape[1]
        layer_sizes = [self.model['W'][0, i].shape[0] for i in range(num_layers)]
        scales = [np.ones(s, dtype=np.float32) for s in layer_sizes]

        if len(self.repair_neural_list):

            for idx, (l, j) in enumerate(sorted(self.repair_neural_list, key=lambda t: t[0])):
                scales[l][j] *= (1.0 + np.float32(r_weight[idx]))


        for i in range(num_layers):
            W = self.model['W'][0, i].astype(np.float32, copy=False)
            b = self.model['b'][0, i].astype(np.float32, copy=False).ravel()
            X = X @ W.T + b

            X *= scales[i]
            if i < num_layers - 1:
                np.maximum(X, 0, out=X)
        return X if x.ndim == 2 and x.shape[1] != 1 else X.T

    def lrp_epsilon(self, input_sample, target_output_index, epsilon=1e-6):

        activations = [input_sample]
        x = input_sample.copy()
        num_layers = self.model['W'].shape[1]


        for i in range(num_layers):
            weights = self.model['W'][0, i]
            biases = self.model['b'][0, i].ravel()
            x = np.dot(weights, x) + biases
            if i < num_layers - 1:
                x = relu(x)
            activations.append(x)

        R = [np.zeros_like(a) for a in activations]
        R[-1][target_output_index] = activations[-1][target_output_index]


        for l in reversed(range(1, len(activations))):
            weights = self.model['W'][0, l - 1]
            biases = self.model['b'][0, l - 1].ravel()
            A = activations[l - 1]
            Z = np.dot(weights, A) + biases + epsilon
            S = R[l] / Z
            C = np.dot(weights.T, S)
            R[l - 1] = A * C

        return R

    def solve_safety(self, beta=0.1, lambda1=0.9, lambda2=0.1, min_source_layer=1):
        print('Repair with contrastive LRP using Top-α strategy and relevance-based source detection:')

        remaining_cex = self.neg_data[0].copy()
        remaining_cex_outputs = self.neg_data[1].copy()

        iteration = 0

        start_time = time.time()

        while len(remaining_cex) > 0 and iteration < self.max_iterations:
            print(f"\n Iteration {iteration + 1}: {len(remaining_cex)} remaining counterexamples")

            paths = []
            path_to_delta = {}

            for i, cex in tqdm(enumerate(remaining_cex), total=len(remaining_cex), desc="Computing paths"):
                pred = self.get_output(cex.reshape(1, -1).T).flatten()
                pred_label = np.argmin(pred)
                correct_neurons = [0]
                actual_output = remaining_cex_outputs[i]
                diffs = [np.abs(actual_output[pred_label] - actual_output[c]) for c in correct_neurons]
                target_label = correct_neurons[np.argmin(diffs)]

                R_wrong = self.lrp_epsilon(cex, target_output_index=pred_label)
                R_correct = self.lrp_epsilon(cex, target_output_index=target_label)
                candidate_set = []
                scores = {}

                L = len(R_wrong) - 2

                for l, (rw, rt) in enumerate(zip(R_wrong[1:-1], R_correct[1:-1])):
                    delta = np.abs(rw - rt).flatten()
                    num_neurons = len(delta)
                    k = max(1, int(beta * num_neurons))
                    top_indices = np.argsort(delta)[-k:]
                    dsd_values = delta[top_indices]
                    dsd_min = np.min(dsd_values)
                    dsd_max = np.max(dsd_values)
                    norm_dsd = (dsd_values - dsd_min) / (dsd_max - dsd_min + 1e-8)
                    pos_score = 1 - l / (L - 1)

                    for idx, j in enumerate(top_indices):
                        phi = lambda1 * norm_dsd[idx] + lambda2 * pos_score
                        candidate_set.append((l, j))
                        scores[(l, j)] = phi

                if not candidate_set:
                    continue
                if min_source_layer > 0:
                    candidate_set = [(l, j) for (l, j) in candidate_set if l >= min_source_layer]

                if not candidate_set:
                    continue


                source = max(candidate_set, key=lambda key: scores[key])
                source_layer = source[0]


                path = []
                for l in range(source_layer, L):
                    layer_candidates = [(l_, j_) for (l_, j_) in candidate_set if l_ == l]
                    sorted_layer = sorted(layer_candidates, key=lambda key: scores[key], reverse=True)
                    path.extend(sorted_layer)

                path_tuple = tuple(path)
                paths.append(path_tuple)
                if path_tuple not in path_to_delta:
                    path_to_delta[path_tuple] = [scores[p] for p in path]

            if not paths:
                print(" 无反例路径，终止")
                break

            path_counter = Counter(paths)
            most_common_path, _ = path_counter.most_common(1)[0]
            deltas = path_to_delta[most_common_path]
            mean_delta = np.mean(deltas)
            print("均值为:", mean_delta)

            print(f" Most common path: {most_common_path}")
            print(f" 源头评分列表: {deltas}")

            for repair_idx in range(len(most_common_path)):
                layer_idx, neuron_idx = most_common_path[repair_idx]
                self.repair_neural_list.append((layer_idx, neuron_idx))
                self.repair_neural_num = len(self.repair_neural_list)
                print(f"Repair neural num: {self.repair_neural_num}")
                print(f"Repair neural list: {self.repair_neural_list}")

                best_cost, best_pos = self.repair()
                self.best_pos = best_pos

                print(f" Repaired neuron fc{layer_idx + 1}-{neuron_idx} | Best cost: {best_cost:.4f}")
                if best_cost < self.threshold:
                    print(" 修复成功，提前结束当前路径")
                    break

            # 训练集评估
            r_safety, r_acc = self.net_accuracy_test(self.pos_data[0], self.neg_data[0], self.best_pos)
            ori_safety, ori_accuracy = self.net_accuracy_test(self.pos_data[0], self.neg_data[0], [])

            print("***********************************************************")
            print("训练集上的结果：")
            print(f' 修复后歧视率: {r_safety} | 准确率: {r_acc}')
            print(f' 修复前歧视率: {ori_safety} | 准确率: {ori_accuracy}')
            self.RSR = (ori_safety - r_safety) /ori_safety
            print(f"训练集上的修复率：{(ori_safety - r_safety) /ori_safety },准确率下降率：{ori_accuracy - r_acc}")

            # 测试集评估
            r_tsafety, r_tacc = self.net_accuracy_test(self.safe_data[0], self.unsafe_data[0], self.best_pos)
            ori_tsafety, ori_taccuracy = self.net_accuracy_test(self.safe_data[0], self.unsafe_data[0], [])

            print("***********************************************************")
            print("测试集上的结果：")
            print(f' 修复后歧视率: {r_tsafety} | 准确率: {r_tacc}')
            print(f' 修复前歧视率: {ori_tsafety} | 准确率: {ori_taccuracy}')
            print(f"测试集上的修复率：{(ori_tsafety - r_tsafety) / ori_tsafety},准确率下降率：{ori_taccuracy - r_tacc}")
            self.GEN = (ori_tsafety - r_tsafety) / ori_tsafety
            self.DD = ori_taccuracy - r_tacc
            if (ori_taccuracy - r_tacc) <= 0.0005 and (ori_tsafety - r_tsafety) / ori_tsafety >= 0.9995:
                print(" 整体修复效果达标，提前终止")
                break

            iteration += 1
        end_time = time.time()
        self.Time = end_time - start_time
        self.NC = self.repair_neural_num
        print(f"all time :{self.Time}")
        return self.NC,self.RSR,self.GEN,self.DD,self.Time


    def repair(self):

        print('Start reparing...')

        options = {'c1': 0.41, 'c2': 0.41, 'w': 0.8}  # parameter tuning
        optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=self.repair_neural_num, options=options,
                                            bounds=([[-1.0] * self.repair_neural_num, [1.0] * self.repair_neural_num]),
                                            init_pos=np.zeros((20, self.repair_neural_num), dtype=float), ftol=1e-3,
                                            ftol_iter=3)


        best_cost, best_pos = optimizer.optimize(self.pso_fitness_func, iters=100)

        return best_cost, best_pos

    def pso_fitness_func(self, weight):

        result = []
        for i in range(0, int(len(weight))):
            r_weight = weight[i]

            safety, accuracy = self.net_accuracy_test(self.pos_data[0], self.neg_data[0], r_weight)

            _result = (1.0 - self.alpha) * safety + self.alpha * (1.0 - accuracy)


            result.append(_result)

        return result

    def net_accuracy_test(self, right_data, wrong_data, r_weight=[]):
        REPAIR_flag = (len(r_weight) != 0)

        if REPAIR_flag:
            out_pos = self.model_repair(right_data, r_weight)
            out_neg = self.model_repair(wrong_data, r_weight)
        else:
            out_pos = self.get_output(right_data)
            out_neg = self.get_output(wrong_data)

        purify_pos_data = self.purify_data([right_data, out_pos])
        purify_neg_data = self.purify_data([wrong_data, out_neg])

        acc = (purify_pos_data[0].shape[0] / right_data.shape[0])
        safety = (wrong_data.shape[0] - purify_neg_data[0].shape[0]) / wrong_data.shape[0]
        return safety, acc