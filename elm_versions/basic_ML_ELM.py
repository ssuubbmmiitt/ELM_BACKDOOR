# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:31:09 2018

@author: admin
"""
from elm_versions.basic_ELM_AE import ELM_AE, ELM_AE_with_mask
from elm_versions.basic_elm import sigmoid, softmax
import numpy as np


def ML_ELM_train(X_train, Y_train, hid_num, k, C):
    betahat_1, param1 = ELM_AE(X_train, hid_num[0], C[0])
    H_temp = np.dot(X_train, betahat_1.T)
    H_1 = sigmoid(H_temp)
    betahat_2, param2 = ELM_AE(H_1, hid_num[1], C[1])
    H_2 = sigmoid(np.dot(H_1, betahat_2.T))
    betahat_3, param3 = ELM_AE(H_2, hid_num[2], C[2])
    H_3 = sigmoid(np.dot(H_2, betahat_3.T))
    H3th3 = np.dot(H_3.T, H_3)
    one_matrix = np.identity(H3th3.shape[0])
    one_matrix = one_matrix * 1 / C[1]
    new_H = H3th3 + one_matrix
    inverse_H = np.linalg.pinv(new_H)
    betahat_4 = np.dot(np.dot(inverse_H, H_3.T), Y_train)
    Y = np.dot(H_3, betahat_4)
    Y = (Y)
    params = (param1, param2, param3)
    return betahat_1, betahat_2, betahat_3, betahat_4, Y, params


# %%
def ML_ELM_test(X, Y, betahat_1, betahat_2, betahat_3, betahat_4, k):
    H_temp = np.dot(X, betahat_1.T)
    H_1 = sigmoid(H_temp)
    H_2 = sigmoid(np.dot(H_1, betahat_2.T))
    H_3 = sigmoid(np.dot(H_2, betahat_3.T))
    Y_predict = np.dot(H_3, betahat_4)
    Y_predict = (Y_predict)
    return Y_predict


def ML_ELM_train_with_mask(X_train, Y_train, hid_num, k, C, prune_rate, params):
    betahat_1 = ELM_AE_with_mask(X_train, hid_num[0], C[0], params[0])
    H_temp = np.dot(X_train, betahat_1.T)
    H_1 = sigmoid(H_temp)
    betahat_2 = ELM_AE_with_mask(H_1, hid_num[1], C[1], params[1])
    H_2 = sigmoid(np.dot(H_1, betahat_2.T))
    betahat_3 = ELM_AE_with_mask(H_2, hid_num[2], C[2], params[2])
    H_3 = sigmoid(np.dot(H_2, betahat_3.T))
    prune_mask = calculate_mask(H_3, prune_rate)
    H3th3 = np.dot(H_3.T, H_3)
    one_matrix = np.identity(H3th3.shape[0])
    one_matrix = one_matrix * 1 / C[1]
    new_H = H3th3 + one_matrix
    inverse_H = np.linalg.pinv(new_H)
    betahat_4 = np.dot(np.dot(inverse_H, H_3.T), Y_train)
    Y = np.dot(H_3, betahat_4)
    Y = (Y)
    return betahat_1, betahat_2, betahat_3, betahat_4, Y, prune_mask


def ML_ELM_test_with_mask(X, Y, betahat_1, betahat_2, betahat_3, betahat_4, k, prune_mask):
    H_temp = np.dot(X, betahat_1.T)
    H_1 = sigmoid(H_temp)
    H_2 = sigmoid(np.dot(H_1, betahat_2.T))
    H_3 = sigmoid(np.dot(H_2, betahat_3.T))
    H_3 = H_3 * prune_mask
    Y_predict = np.dot(H_3, betahat_4)
    Y_predict = (Y_predict)
    return Y_predict


def calculate_mask(h, prune_rate):
    mean = np.mean(h, axis=0)
    prune_mask = np.ones_like(mean)
    number_to_prune = int(prune_rate * len(mean))
    mask_indices = np.argpartition(mean, number_to_prune)[:number_to_prune]
    prune_mask[mask_indices] = 0
    return prune_mask