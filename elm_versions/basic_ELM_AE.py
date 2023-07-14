# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:30:04 2018

@author: admin
"""
from elm_versions.basic_elm import ELM_train, ELM_train_with_mask


def ELM_AE(X_train, hid_num, C):
    W, Beta_hat, Y, parameter = ELM_train(X_train, X_train, hid_num, C)
    return Beta_hat, parameter

def ELM_AE_with_mask(X_train, hid_num, C, param):
    W, Beta_hat, Y = ELM_train_with_mask(X_train, X_train, hid_num, C, param)
    return Beta_hat
