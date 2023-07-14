# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:45:20 2018

@author: Hamid
"""
from elm_versions.basic_TELM import T_ELM_Train, T_ELM_Test, T_ELM_Train_with_mask, T_ELM_Test_with_mask
from sklearn.model_selection import KFold
from elm_versions.predict import predict_new, convert_to_one_hot
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


def TELM_main(X_train, Y_train, X_test, Y_test, hidden_size):
    accuracy_test = np.zeros((1))
    accuracy_train = np.zeros((1))
    n_hid = hidden_size  # L_M[np.argmax(pred_chain)]
    C = 10 ** 6  # C[np.argmax(pred_chain)]
    Wie, Whe, Beta_new, param = None, None, None, None
    import time
    elapsed_time = None

    for i in range(1):
        # print(i)
        start_time = time.time()
        Wie, Whe, Beta_new, param = T_ELM_Train(X_train, Y_train, n_hid, C)
        elapsed_time = time.time() - start_time
        Y_predict_test = T_ELM_Test(X_test, Wie, Whe, Beta_new)
        Y_predict_train = T_ELM_Test(X_train, Wie, Whe, Beta_new)
        accuracy_test[i] = predict_new(Y_test, Y_predict_test)
        accuracy_train[i] = predict_new(Y_train, Y_predict_train)
    final_acc_test = np.sum(accuracy_test) / 1
    final_acc_train = np.sum(accuracy_train) / 1
    final_standard_div = np.sum((accuracy_test - final_acc_test) ** 2) / 1
    stop = time.time()
    # return final_acc_test,final_acc_train,stop-start,final_standard_div
    return final_acc_test, final_acc_train, (Wie, Whe, Beta_new), elapsed_time, param


def TELM_main_with_mask(X_train, Y_train, X_test, Y_test, hidden_size, prune_rate, param):
    accuracy_test = np.zeros((1))
    accuracy_train = np.zeros((1))
    n_hid = hidden_size  # L_M[np.argmax(pred_chain)]
    C = 10 ** 6  # C[np.argmax(pred_chain)]
    Wie, Whe, Beta_new, prune_mask = None, None, None, None
    import time
    elapsed_time = None

    for i in range(1):
        # print(i)
        start_time = time.time()
        Wie, Whe, Beta_new, prune_mask = T_ELM_Train_with_mask(X_train, Y_train, n_hid, C, prune_rate, param)
        elapsed_time = time.time() - start_time
        Y_predict_test = T_ELM_Test_with_mask(X_test, Wie, Whe, Beta_new, prune_mask)
        Y_predict_train = T_ELM_Test_with_mask(X_train, Wie, Whe, Beta_new, prune_mask)
        accuracy_test[i] = predict_new(Y_test, Y_predict_test)
        accuracy_train[i] = predict_new(Y_train, Y_predict_train)
    final_acc_test = np.sum(accuracy_test) / 1
    final_acc_train = np.sum(accuracy_train) / 1
    # return final_acc_test,final_acc_train,stop-start,final_standard_div
    return final_acc_test, final_acc_train, (Wie, Whe, Beta_new), elapsed_time, prune_mask


def TELM_test(X_test, Y_test, Wie, Whe, Beta_new):
    Y_predict_test = T_ELM_Test(X_test, Wie, Whe, Beta_new)
    accuracy_test = predict_new(Y_test, Y_predict_test)
    return accuracy_test


def TELM_test_with_mask(X_test, Y_test, Wie, Whe, Beta_new, prune_mask):
    Y_predict_test = T_ELM_Test_with_mask(X_test, Wie, Whe, Beta_new, prune_mask)
    accuracy_test = predict_new(Y_test, Y_predict_test)
    return accuracy_test