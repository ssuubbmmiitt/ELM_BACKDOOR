import numpy as np
import torch
# import tensorflow as tf
from dataset_handler.mnist import get_alldata_simple
from dataset_handler.trigger import toonehottensor

# %%
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(type(x_train))
# print(x_train.shape)
# print(type(y_train))
# print(y_train.shape)
# x_train = np.reshape(x_train, newshape=(x_train.shape[0], -1))
# x_test = np.reshape(x_test, newshape=(x_test.shape[0], -1))
# print(type(x_train))
# print(x_train.shape)
all_data = get_alldata_simple()
x_train = all_data['train']['x']
y_train = all_data['train']['y']
y_train_oh = all_data['train']['y_oh'].numpy()
x_test = all_data['test']['x']
y_test = all_data['test']['y']
y_test_oh = all_data['test']['y_oh'].numpy()

print(type(x_train))
print(x_train.shape)


# %%
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y




y_train1, y_test1 = toonehottensor(10, y_train.type(torch.LongTensor)).numpy(), toonehottensor(10, y_test.type(torch.LongTensor)).numpy()
y_train_2, y_test_2 = convert_to_one_hot(y_train, 10).T, convert_to_one_hot(y_test, 10).T


print(type(y_train1))
print(y_train1.shape)
print(type(y_train_2))
print(y_train_2.shape)

from elm_versions.DRELM_main import DRELM_main
from elm_versions.ML_ELM_main import main_ML_ELM
from elm_versions.TELM_Main import TELM_main

# acc_train_mnist, acc_test_mnist, final_standard_div_mnist = DRELM_main(x_train, y_train_oh, x_test, y_test_oh)
# acc_train_mnist, acc_test_mnist, final_standard_div_mnist = main_ML_ELM(x_train, y_train_oh, x_test, y_test_oh)
acc_test_mnist, acc_train_mnist = TELM_main(x_train, y_train_oh, x_test, y_test_oh)
print(acc_train_mnist, acc_test_mnist)



