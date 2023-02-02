from util.data import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import torch

torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=50)
from torch.utils.tensorboard import SummaryWriter

def get_full_err_scores(test_result, test_neighbor_result):
    np_test_result = np.array(test_result)
    np_test_neighbor_result = np.array(test_neighbor_result)

    all_scores = None
    all_scores_abs = None
    all_scores_future = None
    all_scores_history = None

    feature_num = np_test_result.shape[-1]

    for i in range(feature_num):
        test_re_list = np_test_result[:2, :, i]
        test_spatioerr = np_test_result[3, :, i]

        test_future_list = np_test_neighbor_result[:2, :, i, :]
        test_history_list = np_test_neighbor_result[2:, :, i, :]

        scores_current = get_err_scores(test_re_list)
        if i in range(0,5):
            test_predict, test_gt = test_re_list
            data_list1 = []
            data_list2 = []
            score = []
            for j in range(11900, 16900):
                data_list1.append(test_predict[j])
                data_list2.append(test_gt[j])
                score.append(scores_current[j])
            x = list(range(0,5000))
            arr1 = np.array(data_list1)
            arr2 = np.array(data_list2)
            arr3 = np.array(score)

            plt.plot(x, arr1, color='green')
            plt.plot(x, arr2, color='black')
            plt.plot(x, arr3, color='red')

            plt.savefig("save_pics/pre_trg_score-{}.png".format(i))
            plt.close()


        scores_current_abs = get_abs_err_scores(test_re_list)
        scores_spatio = test_spatioerr

        f = open('./loss/scores_spatio.txt', 'a')
        for timestamp in range(len(test_spatioerr)):
            print(timestamp, scores_spatio[timestamp], file=f)
        f.close()

        scores_future = get_max_err_scores(test_future_list)
        scores_history = get_max_err_scores(test_history_list)

        scores = scores_current

        if all_scores is None:
            all_scores = scores
        else:
            all_scores = np.vstack((
                all_scores,
                scores
            ))
        if all_scores_abs is None:
            all_scores_abs = scores_current_abs
            all_scores_abs = np.vstack((
                all_scores_abs,
                scores_current_abs
            ))
        if all_scores_future is None:
            all_scores_future = scores_future
        else:
            all_scores_future = np.vstack((
                all_scores_future,
                scores_future
            ))
        if all_scores_history is None:
            all_scores_history = scores_history

        else:
            all_scores_history = np.vstack((
                all_scores_history,
                scores_history
            ))

    return all_scores, all_scores_abs, all_scores_future, all_scores_history


def get_max_err_scores(test_res):
    test_predict, test_gt = test_res

    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64)
    ))

    max_test_delta = test_delta.max(axis=1)


    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)
    epsilon = 1e-2

    err_scores = (max_test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)

    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])


    return smoothed_err_scores



def get_err_scores(test_res):
    test_predict, test_gt = test_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = torch.zeros(test_predict.shape)
    for i in range(len(test_predict)):
        test_delta[i] = np.sqrt((test_predict[i] - test_gt[i]) ** 2)
    q75, q25 = np.percentile(test_delta, [75, 25])
    iqr = q75 - q25
    median = np.median(test_delta)
    a_score = (test_delta - median) / (1 + iqr)

    return a_score

def get_abs_err_scores(test_res):
    test_predict, test_gt = test_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)
    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64)
    ))

    epsilon = 1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)
    smoothed_err_scores = np.zeros(err_scores.shape)

    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

    return smoothed_err_scores

def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):

    total_features = total_err_scores.shape[0]
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]
    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)
    thresold = np.max(normal_scores)
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)


    return f1, pre, rec, auc_score, thresold



def get_best_performance_data(test_result,total_err_scores, total_err_scores_abs_delta, future_scores, history_scores, gt_labels, degree_anamoly_label, topk=1,
                              alpha=0.6, beta=0.1, config={},group_index=1,index=1):
    total_features = total_err_scores.shape[0]
    index_list = []

    topk_indices_future = np.argpartition(future_scores, range(total_features - topk - 1, total_features), axis=0)[
                          -topk:]
    future_topk_err_scores = np.mean(np.take_along_axis(future_scores, topk_indices_future, axis=0), axis=0)
    topk_indices_history = np.argpartition(history_scores, range(total_features - topk - 1, total_features), axis=0)[
                           -topk:]
    history_topk_err_scores = np.mean(np.take_along_axis(history_scores, topk_indices_history, axis=0), axis=0)

    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[
                   -topk:]

    total_topk_err_scores = np.mean(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    for i in range(topk_indices.shape[1]):
        index_list.append(topk_indices[0, i])

    final_topk_fmeas, thresolds = eval_scores_f1(total_topk_err_scores, gt_labels, 800, return_thresold=True,
                                                 win_len=config['slide_win'], down=config['down'])
    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]

    final_topk_fmeas_future, thresolds_future = eval_scores_fh(future_topk_err_scores, gt_labels, 800,
                                                               return_thresold=True)
    th_i_future = final_topk_fmeas_future.index(max(final_topk_fmeas_future))
    thresold_future = thresolds_future[th_i_future]

    final_topk_fmeas_history, thresolds_history = eval_scores_fh(history_topk_err_scores, gt_labels, 800,
                                                                 return_thresold=True)
    th_i_history = final_topk_fmeas_history.index(max(final_topk_fmeas_history))
    thresold_history = thresolds_history[th_i_history]


    final_topk_fmeas_future_f1, thresolds_future_f1 = eval_scores_f1(future_topk_err_scores, gt_labels, 800,
                                                                     return_thresold=True,win_len=config['slide_win'],down=config['down'])
    th_i_future_f1 = final_topk_fmeas_future_f1.index(max(final_topk_fmeas_future_f1))
    thresold_future_f1 = thresolds_future_f1[th_i_future_f1]

    final_topk_fmeas_history_f1, thresolds_history_f1 = eval_scores_f1(history_topk_err_scores, gt_labels, 800,
                                                                       return_thresold=True,win_len=config['slide_win'],down=config['down'])
    th_i_history_f1 = final_topk_fmeas_history_f1.index(max(final_topk_fmeas_history_f1))
    thresold_history_f1 = thresolds_history_f1[th_i_history_f1]


    pred_labels_0 = np.zeros(len(total_topk_err_scores))
    pred_labels = np.zeros(len(total_topk_err_scores))

    pred_labels_0[total_topk_err_scores > thresold] = 1
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels_0)):
        if pred_labels_0[i] == 0 and degree_anamoly_label[i] == 1:
            pred_labels[i] = 1


    win_len = config['slide_win']

    anomaly_time = pd.read_csv('../data/swat/SWAT_Time.csv')

    anomaly_time = np.array(anomaly_time.iloc[:, :2])
    anomaly_start = anomaly_time[:, 0]
    anomaly_end = anomaly_time[:, 1]
    down = config['down']

    for i in range(len(anomaly_start)):
        if np.sum(pred_labels_0[int(anomaly_start[i] / down) - win_len:int(
                anomaly_end[i] / down) - win_len + 1]) > 0:
            pred_labels_0[int(anomaly_start[i] / down) - win_len:int(anomaly_end[i] / down) - win_len] = 1

    for i in range(len(anomaly_start)):
        if np.sum(pred_labels[int(anomaly_start[i] / down) - win_len:int(
                anomaly_end[i] / down) - win_len + 1]) > 0:
            pred_labels[int(anomaly_start[i] / down) - win_len:int(anomaly_end[i] / down) - win_len] = 1

    for i in range(len(pred_labels)):
        pred_labels_0[i] = int(pred_labels_0[i])
        pred_labels[i] = int(pred_labels[i])


    pre1 = precision_score(gt_labels, pred_labels_0, zero_division=1)
    rec1 = recall_score(gt_labels, pred_labels_0)
    f11 = f1_score(gt_labels, pred_labels_0)


    pre = precision_score(gt_labels, pred_labels, zero_division=1)
    rec = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)


    print('pred_current_MSE f11, pre1, rec1', f11, pre1, rec1, thresold)
    print('pred_current_MSE graph,f1, pre, rec', f1, pre, rec, thresold)

    f = open('./loss/f1.txt', 'a')
    print('***********group_index,index', group_index, index, file=f)
    print('pred_current_MSE f11, pre1, rec1', f11, pre1, rec1, thresold, file=f)
    print('pred_current_MSE graph,f1, pre, rec', f1, pre, rec, thresold, file=f)
    f.close()

    auc_score =0
    return f1, pre, rec, auc_score, thresold
