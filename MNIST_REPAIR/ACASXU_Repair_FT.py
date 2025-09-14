#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MuXiaRepair
@File    ：ACASXU_Repair_FT
@Author  ：慕夏
@Date    ：2025/7/15
'''

import time
import copy
import h5py
import numpy as np
from tqdm import tqdm
import pyswarms as ps
from collections import Counter, defaultdict
import datetime


def relu(x: np.ndarray) -> np.ndarray:

    return np.maximum(x, 0.0, dtype=x.dtype)


class SPGR_REPAIR_FT:

    def __init__(self, model, repair_num, alpha, max_iterations):
        self.model = model
        self.model_R = copy.deepcopy(model)
        self.pso_particle_batch = 8
        self.pso_data_batch = 2048


        self.W_list = [np.ascontiguousarray(model['W'][0, i]).astype(np.float32, copy=False)
                       for i in range(model['W'].shape[1])]
        self.b_list = [np.ravel(model['b'][0, i]).astype(np.float32, copy=False)
                       for i in range(model['b'].shape[1])]

        self.W_list_R = [np.ascontiguousarray(self.model_R['W'][0, i]).astype(np.float32, copy=False)
                         for i in range(self.model_R['W'].shape[1])]

        self.b_list_R = [np.ravel(self.model_R['b'][0, i]).astype(np.float32, copy=False)
                         for i in range(self.model_R['b'].shape[1])]

        self.num_layers = len(self.W_list)

        self.repair_num = repair_num
        self.alpha = float(alpha)
        self.max_iterations = int(max_iterations)


        self.pos_train_data = []
        self.neg_train_data = []
        self.fog_test_data = []
        self.ide_test_data = []


        self.repair_neural_num = 1
        self.repair_neural_list = []
        self.threshold = 1e-5
        self.best_pos = 0.0

    def get_data(self, nn):

        base = "../benchmarks/"
        def load_csv(path, dtype=np.float32):
            return np.loadtxt(path, delimiter=',', dtype=dtype)

        print("load positive inputs...")
        pos_in = load_csv(base + nn + "/positive_inputs.csv")
        pos_in = np.ascontiguousarray(pos_in, dtype=np.float32)

        print("load positive outputs...")
        pos_out = load_csv(base + nn + "/pos_outputs.csv")

        pos_out = np.argmax(pos_out, axis=1).astype(np.int64, copy=False)

        print("load negative inputs...")
        neg_in = load_csv(base + nn + "/negative_inputs.csv")
        print(len(neg_in))
        neg_in = np.ascontiguousarray(neg_in, dtype=np.float32)

        print("load negative outputs...")
        cor_out = load_csv(base + nn + "/cor_outputs.csv")
        cor_out = np.argmax(cor_out, axis=1).astype(np.int64, copy=False)


        print("load fog_Generalization inputs...")
        fog_test_in = np.load(base + "/fog_Generalization/test_images.npy").astype(np.float32)
        print(len(fog_test_in))
        fog_test_in = fog_test_in.reshape((-1, 28 * 28))
        fog_test_in *= (1.0 / 255.0)

        print("load fog_Generalization outputs...")
        fog_test_out = np.load(base + "/fog_Generalization/test_labels.npy").astype(np.int64)

        print("load identity_Drawdown inputs...")
        ide_test_in = np.load(base + "/identity_Drawdown/test_images.npy").astype(np.float32)
        print(len(ide_test_in))
        ide_test_in = ide_test_in.reshape((-1, 28 * 28))
        ide_test_in *= (1.0 / 255.0)

        print("load identity_Drawdown outputs...")
        ide_test_out = np.load(base + "/identity_Drawdown/test_labels.npy").astype(np.int64)

        self.pos_train_data = [pos_in, pos_out]
        self.neg_train_data = [neg_in, cor_out]
        self.fog_test_data = [fog_test_in, fog_test_out]
        self.ide_test_data = [ide_test_in, ide_test_out]
        return True

    def _build_layer_gains_batch(self, weights_batch: np.ndarray):

        B = weights_batch.shape[0]
        layer_out_dims = [w.shape[0] for w in self.W_list_R]
        gains = [np.ones((B, d), dtype=np.float32) for d in layer_out_dims]


        sorted_pairs = sorted(self.repair_neural_list, key=lambda x: x[0])

        for col, (l, j) in enumerate(sorted_pairs):
            gains[l][:, j] *= (1.0 + weights_batch[:, col])
        return gains

    def _forward_with_gains_batch(self, x_chunk: np.ndarray, gains_list):

        B = gains_list[0].shape[0]
        M = x_chunk.shape[0]


        out = np.repeat(x_chunk[None, :, :], B, axis=0).astype(np.float32, copy=False)

        for l in range(self.num_layers):
            in_dim = out.shape[2]
            out_dim = self.W_list_R[l].shape[0]


            tmp = out.reshape(B * M, in_dim) @ self.W_list_R[l].T
            tmp += self.b_list_R[l]
            tmp = np.maximum(tmp, 0.0, dtype=tmp.dtype)
            out = tmp.reshape(B, M, out_dim)


            out *= gains_list[l][:, None, :]

        return out  # (B, M, C)


    def _forward_activations(self, x: np.ndarray):

        acts = [np.ascontiguousarray(x, dtype=np.float32)]
        a = acts[0]
        W_list, B_list = self.W_list_R, self.b_list_R
        for l in range(self.num_layers):
            a = relu(a @ W_list[l].T + B_list[l])  # (out_dim,)
            acts.append(a)
        return acts

    def _lrp_epsilon_dual(self, activations, idx_wrong: int, idx_true: int, epsilon: float = 1e-6):

        W_list, B_list = self.W_list_R, self.b_list_R
        L = self.num_layers  # 隐藏层数（3）


        out_dim = activations[-1].shape[0]
        R = np.zeros((out_dim, 2), dtype=np.float32)
        R[idx_wrong, 0] = activations[-1][idx_wrong]
        R[idx_true, 1] = activations[-1][idx_true]

        Rs = [None] * L

        Rs[L - 1] = R.copy()


        for l in range(L, 0, -1):
            A_prev = activations[l - 1]
            Z = (W_list[l - 1] @ A_prev) + B_list[l - 1] + epsilon
            S = R / Z[:, None]
            C = W_list[l - 1].T @ S
            R_prev = (A_prev[:, None] * C).astype(np.float32, copy=False)


            if l - 1 >= 1:
                hidden_idx = (l - 1) - 1
                Rs[hidden_idx] = R_prev

            R = R_prev

        return Rs

    def get_output(self, x: np.ndarray) -> np.ndarray:

        out = np.ascontiguousarray(x, dtype=np.float32)
        for i in range(self.num_layers):
            out = out @ self.W_list[i].T + self.b_list[i]  # (N, out_dim)
            out = relu(out)
        return out


    def apply_repair_to_model_R(self):

        if not isinstance(self.best_pos, np.ndarray):
            return

        sorted_pairs = sorted(self.repair_neural_list, key=lambda x: x[0])
        for idx, (l, j) in enumerate(sorted_pairs):
            factor = 1.0 + float(self.best_pos[idx])
            self.W_list_R[l][j, :] *= factor



    def _build_layer_gains(self, r_weight: np.ndarray):

        if len(self.repair_neural_list) == 0:
            return None

        sorted_pairs = sorted(self.repair_neural_list, key=lambda x: x[0])
        layer_out_dims = [w.shape[0] for w in self.W_list_R]
        gains = [None] * self.num_layers

        for l in range(self.num_layers):
            gains[l] = np.ones(layer_out_dims[l], dtype=np.float32)

        for idx, (l, j) in enumerate(sorted_pairs):
            gains[l][j] *= (1.0 + float(r_weight[idx]))
        return gains

    def model_repair(self, x: np.ndarray, r_weight) -> np.ndarray:

        out = np.ascontiguousarray(x, dtype=np.float32)
        if isinstance(r_weight, (list, tuple)):
            r_weight = np.asarray(r_weight, dtype=np.float32)
        gains = self._build_layer_gains(r_weight) if len(r_weight) != 0 else None

        for l in range(self.num_layers):
            out = out @ self.W_list_R[l].T + self.b_list_R[l]
            out = relu(out)
            if gains is not None:
                out *= gains[l]
        return out


    def lrp_epsilon(self, input_sample: np.ndarray, target_output_index: int, epsilon: float = 1e-6):

        activations = [np.ascontiguousarray(input_sample, dtype=np.float32)]
        x = activations[0]
        for l in range(self.num_layers):
            x = relu(self.W_list_R[l] @ x + self.b_list_R[l])
            activations.append(x)

        R = [np.zeros_like(a, dtype=np.float32) for a in activations]
        R[-1][target_output_index] = activations[-1][target_output_index]

        for l in range(self.num_layers, 0, -1):
            W = self.W_list_R[l - 1]
            b = self.b_list_R[l - 1]
            A = activations[l - 1]
            # Z = W @ A + b + eps
            Z = W @ A + b + epsilon
            S = R[l] / Z
            C = W.T @ S
            R[l - 1] = A * C
        return R

    def solve_robustness(self, alpha=0.7, lambda1=0.9, lambda2=0.1, repair_rate=0.8):
        print(' 开始鲁棒性修复（全局Top-α + 源头评分 + 路径优先）...')

        remaining_cex = np.ascontiguousarray(self.neg_train_data[0], dtype=np.float32)
        remaining_labels = np.asarray(self.neg_train_data[1], dtype=np.int64)

        iteration = 0
        pre_cost = 1.0
        print("当前时间:", datetime.datetime.now())
        while len(remaining_cex) > 0 and iteration < self.max_iterations:
            print(f"\n Iteration {iteration + 1} | 剩余反例数量: {len(remaining_cex)}")

            path_counter = Counter()
            path_to_cex_indices = defaultdict(list)
            path_to_scores = {}


            all_preds = self.get_output(remaining_cex)
            all_pred_labels = np.argmax(all_preds, axis=1)
            wrong_mask = (all_pred_labels != remaining_labels)
            wrong_indices = np.nonzero(wrong_mask)[0]

            for idx in tqdm(wrong_indices, desc="Computing paths"):
                cex_sample = remaining_cex[idx]
                pred_label = int(all_pred_labels[idx])
                true_label = int(remaining_labels[idx])


                acts = self._forward_activations(cex_sample)
                Rs_dual = self._lrp_epsilon_dual(acts, pred_label, true_label)

                L_hidden = self.num_layers  # 3 隐藏层

                top_alpha = float(alpha)
                lam1, lam2 = float(lambda1), float(lambda2)


                cand_layers = []
                cand_indices = []
                cand_scores = []


                for l in range(L_hidden):
                    Rw = Rs_dual[l][:, 0]  # 错类相关性
                    Rt = Rs_dual[l][:, 1]  # 真类相关性
                    delta = np.abs(Rw - Rt)  # (units,)

                    n = delta.size  # 每层的神经元数量
                    k = max(1, int(top_alpha * n))  # 计算每层的 Top-k，top_alpha=0.1 时就是取前 10 个

                    # 取前 k 个神经元，并按分数降序排列
                    if k < n:
                        part = np.argpartition(delta, -k)[-k:]
                    else:
                        part = np.arange(n, dtype=np.int64)

                    order = part[np.argsort(delta[part])[::-1]]
                    dsd = delta[order]

                    # 归一化
                    dmin, dmax = float(dsd.min()), float(dsd.max())
                    norm = (dsd - dmin) / ((dmax - dmin) + 1e-8)

                    # 位置分数（根据层的深度）
                    pos_score = 1.0 - (l / max(L_hidden - 1, 1))
                    phi = lam1 * norm + lam2 * pos_score

                    cand_layers.append(np.full(phi.shape, l, dtype=np.int16))
                    cand_indices.append(order.astype(np.int32, copy=False))
                    cand_scores.append(phi.astype(np.float32, copy=False))

                if not cand_scores:
                    continue


                cand_layers = np.concatenate(cand_layers, axis=0)
                cand_indices = np.concatenate(cand_indices, axis=0)
                cand_scores = np.concatenate(cand_scores, axis=0)

                # 源头：全局最高分
                src_idx = int(np.argmax(cand_scores))
                source_layer = int(cand_layers[src_idx])

                # 从源头层往后组成路径：每层按分数降序
                path_pairs = []
                for l in range(source_layer, L_hidden):
                    maskL = (cand_layers == l)
                    if not np.any(maskL):
                        continue
                    idxs = np.nonzero(maskL)[0]
                    ordL = idxs[np.argsort(cand_scores[idxs])[::-1]]

                    layer_js = cand_indices[ordL]
                    path_pairs.extend((l, int(j)) for j in layer_js)

                path_tuple = tuple(path_pairs)
                path_counter[path_tuple] += 1
                path_to_cex_indices[path_tuple].append(int(idx))

            if not path_counter:
                print(" 无可用修复路径，终止")
                break


            for path_rank, (path, _) in enumerate(path_counter.most_common()):
                print(f"\n 尝试第 {path_rank + 1} 条路径（反例覆盖数: {path_counter[path]}）")
                print(path)
                print(len(path))
                self.repair_neural_list = list(path)
                self.repair_neural_num = len(self.repair_neural_list)

                best_cost, best_pos = self.repair()
                self.best_pos = best_pos
                print(f" 修复后代价: {best_cost:.6f}")

                pred_cex = self.model_repair(remaining_cex, self.best_pos)
                pred_labels = np.argmax(pred_cex, axis=1)
                fixed_indices = np.where(pred_labels == remaining_labels)[0]
                repaired_rate = len(fixed_indices) / len(remaining_cex)
                print(f" 当前路径修复比例: {repaired_rate:.2%}")

                if repaired_rate >= repair_rate or best_cost < self.threshold:
                    print(" 修复效果达标，终止当前路径修复")
                    if best_cost < pre_cost:
                        self.apply_repair_to_model_R()
                        pre_cost = best_cost
                    break

                if best_cost < pre_cost:
                    self.apply_repair_to_model_R()
                    pre_cost = best_cost

                # 评估（与原逻辑一致）
                for name, (pos_data, neg_data) in {
                    "训练集": (self.pos_train_data, self.neg_train_data),
                    "测试集": (self.ide_test_data, self.fog_test_data)
                }.items():
                    r_acc, r_safety = self.net_accuracy_test(pos_data, neg_data, self.best_pos)
                    ori_acc, ori_safety = self.net_accuracy_test(pos_data, neg_data, [])
                    print(f"[{name}] 修复前歧视率: {ori_safety:.4f} | 修复后: {r_safety:.4f}")
                    print(f"[{name}] 修复前准确率: {ori_acc:.4f} | 修复后: {r_acc:.4f}")
                    print(f"✅ 修复增长: {r_safety - ori_safety:.4f} | 修复后准确率减少: {ori_acc - r_acc:.4f}")
                    print("**************************************************")
                print("当前时间:", datetime.datetime.now())

            # 移除修复成功的反例
            pred_cex = self.model_repair(remaining_cex, self.best_pos)
            pred_labels = np.argmax(pred_cex, axis=1)
            remain_mask = (pred_labels != remaining_labels)
            remaining_cex = remaining_cex[remain_mask]
            remaining_labels = remaining_labels[remain_mask]

            iteration += 1


            r_acc, r_safety = self.net_accuracy_test(self.pos_train_data, self.neg_train_data, self.best_pos)
            ori_accuracy, ori_safety = self.net_accuracy_test(self.pos_train_data, self.neg_train_data, [])
            print(f"✅ 训练修复前歧视率: {ori_safety:.4f} | 训练修复后歧视率: {r_safety:.4f}")
            print(f"✅ 训练修复前准确率: {ori_accuracy:.4f} | 训练修复后准确率: {r_acc:.4f}")
            print(f"✅ 训练修复增长: {r_safety - ori_safety:.4f} | 训练修复后准确率增长: {r_acc - ori_accuracy:.4f}")
            print("**************************************************")

            r_acc, r_safety = self.net_accuracy_test(self.ide_test_data, self.fog_test_data, self.best_pos)
            ori_accuracy, ori_safety = self.net_accuracy_test(self.ide_test_data, self.fog_test_data, [])
            print(f"✅ 修复前歧视率: {ori_safety:.4f} | 修复后歧视率: {r_safety:.4f}")
            print(f"✅ 修复前准确率: {ori_accuracy:.4f} | 修复后准确率: {r_acc:.4f}")
            print(f"✅ 修复增长: {r_safety - ori_safety:.4f} | 修复后准确率增长: {r_acc - ori_accuracy:.4f}")

            # 原条件保持
            if (ori_accuracy - r_acc) <= 0.0005 and (ori_safety - r_safety) / max(ori_safety, 1e-12) >= 0.9995:
                print(" 整体修复效果达标，提前终止")
                break

        return True


    def repair(self):
        print('Start reparing...')
        options = {'c1': 0.41, 'c2': 0.41, 'w': 0.8}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=20,
            dimensions=self.repair_neural_num,
            options=options,
            bounds=([[-1.0] * self.repair_neural_num, [1.0] * self.repair_neural_num]),
            init_pos=np.zeros((20, self.repair_neural_num), dtype=np.float32),
            ftol=1e-3, ftol_iter=10

        )
        best_cost, best_pos = optimizer.optimize(self.pso_fitness_func, iters=100)
        return best_cost, best_pos

    def pso_fitness_func(self, weight: np.ndarray):
        pos_x, pos_y = self.pos_train_data
        neg_x, neg_y = self.neg_train_data

        P, D = weight.shape
        PB = int(self.pso_particle_batch)
        DB = int(self.pso_data_batch)

        acc_pos = np.zeros(P, dtype=np.float64)
        acc_neg = np.zeros(P, dtype=np.float64)


        for p0 in range(0, P, PB):
            p1 = min(p0 + PB, P)
            wb = weight[p0:p1]  # (B,D)
            B = wb.shape[0]
            gains = self._build_layer_gains_batch(wb)
            # 正样本准确率
            correct = np.zeros(B, dtype=np.int64);
            total = 0
            for s in range(0, len(pos_x), DB):
                t = min(s + DB, len(pos_x))
                preds = self._forward_with_gains_batch(pos_x[s:t], gains)  # (B, m, C)
                yhat = np.argmax(preds, axis=2)  # (B, m)

                cmp = (yhat == pos_y[s:t][None, :])
                correct += cmp.sum(axis=1)
                total += (t - s)
            acc_pos[p0:p1] = correct / max(total, 1)

            # 负样本准确率
            correct = np.zeros(B, dtype=np.int64);
            total = 0
            for s in range(0, len(neg_x), DB):
                t = min(s + DB, len(neg_x))
                preds = self._forward_with_gains_batch(neg_x[s:t], gains)  # (B, m, C)
                yhat = np.argmax(preds, axis=2)
                cmp = (yhat == neg_y[s:t][None, :])
                correct += cmp.sum(axis=1)
                total += (t - s)
            acc_neg[p0:p1] = correct / max(total, 1)


        return ((1.0 - self.alpha) * (1.0 - acc_neg) + self.alpha * (1.0 - acc_pos)).astype(np.float32)


    def net_accuracy_test(self, pos_data, neg_data, r_weight=[]):
        REPAIR = (len(r_weight) != 0)
        x_pos, y_pos = pos_data[0], pos_data[1]
        x_neg, y_neg = neg_data[0], neg_data[1]

        if REPAIR:
            pred_pos = self.model_repair(x_pos, r_weight)
            pred_neg = self.model_repair(x_neg, r_weight)
        else:
            pred_pos = self.get_output(x_pos)
            pred_neg = self.get_output(x_neg)

        yhat_pos = np.argmax(pred_pos, axis=1)
        yhat_neg = np.argmax(pred_neg, axis=1)

        acc_pos = float((yhat_pos == y_pos).sum()) / y_pos.size
        acc_neg = float((yhat_neg == y_neg).sum()) / y_neg.size
        return acc_pos, acc_neg

    def net_accuracy_test_prob(self, pos_data, neg_data, r_weight=[]):
        REPAIR = (len(r_weight) != 0)
        x_pos, y_pos = pos_data[0], pos_data[1]
        x_neg, y_neg = neg_data[0], neg_data[1]

        if REPAIR:
            pred_pos = self.model_repair(x_pos, r_weight)
            pred_neg = self.model_repair(x_neg, r_weight)
        else:
            pred_pos = self.get_output(x_pos)
            pred_neg = self.get_output(x_neg)

        pos_dist = np.abs(np.sum(np.linalg.norm(pred_pos - y_pos, axis=1)))
        neg_dist = np.abs(np.sum(np.linalg.norm(pred_neg - y_neg, axis=1)))
        return pos_dist, neg_dist
