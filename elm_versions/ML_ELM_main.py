# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:39:21 2018

@author: admin
"""
import time
# from  dataset import load_mnist
# from dataset import load_sat
# from dataset import load_duke
# from dataset import load_hill_valley,load_olivetti_faces
# from dataset import load_usps,load_cars,load_leaves
import numpy as np
from sklearn.model_selection import KFold
from elm_versions.predict import predict_new, convert_to_one_hot
from elm_versions.basic_ML_ELM import ML_ELM_train, ML_ELM_test, ML_ELM_train_with_mask, ML_ELM_test_with_mask


def main_ML_ELM(X_train, Y_train, X_test, Y_test, hidden_layer: int = 700):

    hddn_lyrs = [200, 200]
    hddn_lyrs.append(hidden_layer)
    accuracy = np.zeros((1))
    n_hid = hddn_lyrs
    CC = [10 ** 6, 10 ** 6, 10 ** 6]
    betahat_1, betahat_2, betahat_3, betahat_4, params = None, None, None, None, None
    elapsed_time = None
    for i in range(1):
        start_time = time.time()
        betahat_1, betahat_2, betahat_3, betahat_4, Y, params = ML_ELM_train(X_train, Y_train, n_hid, 10, CC)
        elapsed_time = time.time() - start_time
        Y_predict = ML_ELM_test(X_test, Y_test, betahat_1, betahat_2, betahat_3, betahat_4, 10)
        accuracy[i] = predict_new(Y_test, Y_predict)
    final_acc = np.sum(accuracy) / 1
    # final_standard_div = np.sum((accuracy - final_acc) ** 2) / 1
    # return final_acc, stop - start, final_standard_div
    return final_acc, (betahat_1, betahat_2, betahat_3, betahat_4), elapsed_time, params




def main_ML_ELM_with_mask(X_train, Y_train, X_test, Y_test, prune_rate, params, hidden_layer: int = 700):

    hddn_lyrs = [200, 200]
    hddn_lyrs.append(hidden_layer)
    accuracy = np.zeros((1))
    n_hid = hddn_lyrs
    CC = [10 ** 6, 10 ** 6, 10 ** 6]
    betahat_1, betahat_2, betahat_3, betahat_4, prune_mask = None, None, None, None, None
    elapsed_time = None
    for i in range(1):
        start_time = time.time()
        betahat_1, betahat_2, betahat_3, betahat_4, Y, prune_mask = ML_ELM_train_with_mask(X_train, Y_train, n_hid, 10, CC, prune_rate, params)
        elapsed_time = time.time() - start_time
        Y_predict = ML_ELM_test_with_mask(X_test, Y_test, betahat_1, betahat_2, betahat_3, betahat_4, 10, prune_mask)
        accuracy[i] = predict_new(Y_test, Y_predict)
    final_acc = np.sum(accuracy) / 1
    final_standard_div = np.sum((accuracy - final_acc) ** 2) / 1
    # return final_acc, stop - start, final_standard_div
    if prune_mask is None:
        raise ValueError("prune_mask is None")
    return final_acc, (betahat_1, betahat_2, betahat_3, betahat_4), elapsed_time, prune_mask



def main_ML_ELM_test(X_train, Y_train, X_test, Y_test, hidden_layer: int = 700):

    hddn_lyrs = [200, 200]
    hddn_lyrs.append(hidden_layer)
    accuracy = np.zeros((1))
    n_hid = hddn_lyrs
    CC = [10 ** 6, 10 ** 6, 10 ** 6]
    betahat_1, betahat_2, betahat_3, betahat_4 = None, None, None, None
    elapsed_time = None
    for i in range(1):
        start_time = time.time()
        betahat_1, betahat_2, betahat_3, betahat_4, Y = ML_ELM_train(X_train, Y_train, n_hid, 10, CC)
        elapsed_time = time.time() - start_time
        Y_predict = ML_ELM_test(X_test, Y_test, betahat_1, betahat_2, betahat_3, betahat_4, 10)
        accuracy[i] = predict_new(Y_test, Y_predict)
    final_acc = np.sum(accuracy) / 1
    final_standard_div = np.sum((accuracy - final_acc) ** 2) / 1
    # return final_acc, stop - start, final_standard_div
    return final_acc, (betahat_1, betahat_2, betahat_3, betahat_4), elapsed_time


def MLELM_test(X_test, Y_test, betahat_1, betahat_2, betahat_3, betahat_4):
    Y_predict = ML_ELM_test(X_test, Y_test, betahat_1, betahat_2, betahat_3, betahat_4, 10)
    accuracy = predict_new(Y_test, Y_predict)

    return accuracy

def MLELM_test_with_mask(X_test, Y_test, betahat_1, betahat_2, betahat_3, betahat_4, prune_mask):
    Y_predict = ML_ELM_test_with_mask(X_test, Y_test, betahat_1, betahat_2, betahat_3, betahat_4, 10, prune_mask)
    accuracy = predict_new(Y_test, Y_predict)

    return accuracy